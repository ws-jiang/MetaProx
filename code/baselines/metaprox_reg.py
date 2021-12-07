import torch
from torch import nn
import numpy as np

from backbones.backbone_utils import get_backbone
from data.dataset_utils import DatasetEnum
from data.get_dataset import get_dataset
from utils.evaluation_utils import Average
from utils.log_utils import LogUtils
from baselines.kernels import KernelZoo as K
from utils.path_utils import PathUtils
from utils.progress_utils import ProgressConfig
from utils.time_utils import TimeCounter
from utils.torch_utils import TorchUtils
from utils.tensor_utils import split_support_query

class LinearProximalModel(nn.Module):
    def __init__(self, feature_dim=40, out_dim=1):
        super(LinearProximalModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, out_dim),
        )

    def forward(self, x):
        y_pred = self.net(x)
        return y_pred


class MetaProxModel(nn.Module):
    def __init__(self, config):
        super(MetaProxModel, self).__init__()
        self.config = config
        self.ds_name = self.config["ds"]
        self.backbone_name = self.config["backbone_name"]
        self.feature_extractor, self.feature_dim = get_backbone(backbone_name=self.backbone_name, ds_name=self.ds_name, config=config)

        self.proximal_model = LinearProximalModel(feature_dim=self.feature_dim)

    def __repr__(self):
        return "{}\t{}".format(self.backbone_name, self.ds_name)

class MetaProxRegression(object):
    def __init__(self, config):
        self.config = config

        self.n_support = config["n_support"]
        self.n_query = config["n_query"]

        self.ds_name = self.config["ds"]
        self.index_key = self.config["index_key"]
        self.job_id = self.config["job_id"]
        self.task_name = "MetaProx_Reg"
        self.logger = LogUtils.get_or_init_logger(dir_name=self.task_name, file_name=self.job_id)

        self.is_out_range = self.config.get("is_out_range", False) # for QMUL
        self.dataset = get_dataset(self.ds_name, split=None, is_out_range=self.is_out_range)

        self.lr = self.config["lr"]

        self.criterion = nn.MSELoss()
        self.noise_sigma = self.config["noise_sigma"]
        self.progress = ProgressConfig(self.config)

    def task_update(self, x_support, y_support, prior_model):
        """
        task update aims to solve the dual problem in the inner loop
        remark: the dual problem has a closed-form solution
        x_support: batch * n_support * d
        y_support: batch * n_support * 1
        """
        n_support = x_support.size(1)
        num_task = x_support.size(0)

        batch_eye = torch.eye(n_support).unsqueeze(0).expand(num_task, -1, -1).cuda()
        kernel_mat = K.compute_linear(x_support, x_support)
        kernel_mat = (kernel_mat + kernel_mat.transpose(1, 2)) / 2  # make symmetric
        c_mat = torch.inverse(batch_eye + kernel_mat)
        alpha = c_mat.bmm(y_support - prior_model(x_support))
        return alpha  # num_task * n_support * 1

    def meta_update(self, x_support, y_support, x_query, y_query, prior_model):
        """
        x_support: batch * n_support * d
        y_support: batch * n_support * 1
        x_query: batch * n_query * d
        y_query: batch * n_query * 1
        x_0: n_0 * d
        """

        alpha = self.task_update(x_support, y_support, prior_model)
        y_pred_prior = prior_model(x_query)  # num_task * n_support * 1
        y_pred_posterior = K.compute_linear(x_query, x_support).bmm(alpha)  # num_task * n_support * 1
        y_pred = y_pred_prior + y_pred_posterior

        loss = self.criterion(y_pred, y_query)
        return loss, y_pred, y_pred_prior

    def get_best_model(self):
        return "{}_{}_{}".format(self.job_id, self.config["index_key"], "best")

    def get_identifier(self):
        return self.job_id

    def sample_tasks(self, n_task, stage):
        x = []
        y = []
        sample_num = self.n_support + self.n_query
        for i in range(n_task):
            task_id = self.dataset.sample_a_task(stage=stage)
            x_, y_ = self.dataset.sample_examples(task_id=task_id, num_samples=sample_num, stage=stage)
            x.append(x_)
            y.append(y_)

        return torch.stack(x).cuda(), torch.stack(y).cuda()

    def get_num_steps(self, stage):
        if self.ds_name == DatasetEnum.QMUL.name:
            return 100
        else:
            return self.config["num_tasks"][stage]

    def eval(self, model=None, stage="valid", epoch=0):
        torch.set_grad_enabled(False)
        model.train(False)

        # QMUL no validation set, thus, no need to reload model
        if stage == "test" and self.ds_name != DatasetEnum.QMUL.name:
            model = MetaProxModel(config=self.config)
            model.cuda()
            checkpoint = PathUtils.load_ckp(dir_name=self.task_name, identifier=self.get_identifier(), index_key=self.index_key)
            model.load_state_dict(checkpoint["model_state_dict"])
        stat_keys = ["loss"]
        avg_dict = {x: Average(x) for x in stat_keys}
        tc = TimeCounter()
        num_steps = self.get_num_steps(stage)
        batch_task_num = self.config["batch_task_num"]
        for step_iter in range(1, num_steps + 1):
            x, y = self.sample_tasks(n_task=batch_task_num, stage=stage)
            y = y + torch.rand_like(y) * self.noise_sigma
            x = model.feature_extractor.forward_feature(x.view(x.shape[0] * x.shape[1], *x.shape[2:])).view(x.shape[0], x.shape[1], -1)
            x_support, y_support, x_query, y_query = split_support_query(x, y, n_support=self.n_support)
            loss, _, _ = self.meta_update(x_support, y_support, x_query, y_query, model.proximal_model)

            avg_dict["loss"].add(loss.item())

        msg = "{} {} {}: {}  {}, cost: {} ms".format(self.job_id, epoch, stage.upper(), self.job_id, LogUtils.get_stat_from_dict(avg_dict), tc.count_ms())
        self.logger.info(msg)
        torch.set_grad_enabled(True)
        model.train(True)
        return avg_dict

    def train(self, seed=319):

        stat_keys = ["loss", "ts"]
        current_best_score = float("inf")
        not_improve_cnt = 0

        TorchUtils.set_random_seed(seed)
        model = MetaProxModel(config=self.config)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        stage = "train"
        batch_task_num = self.config["batch_task_num"]
        num_tasks = self.get_num_steps(stage)

        avg_dict = {x: Average(x) for x in stat_keys}
        for i_iter in range(1, num_tasks+1):
            if not_improve_cnt >= self.progress.max_not_impr_cnt:
                break

            torch.set_grad_enabled(True)

            x, y = self.sample_tasks(n_task=batch_task_num, stage=stage)
            y = y + torch.rand_like(y) * self.noise_sigma
            x = model.feature_extractor.forward_feature(x.view(x.shape[0] * x.shape[1],
                                                               *x.shape[2:])).view(x.shape[0], x.shape[1], -1)
            x_support, y_support, x_query, y_query = split_support_query(x, y, n_support=self.n_support)

            loss, y_pred, _ = self.meta_update(x_support, y_support, x_query, y_query, prior_model=model.proximal_model)
            avg_dict["loss"].add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            msg = "{} {}: {}/{}\t{}".format(self.job_id, stage.upper(), i_iter,
                                            num_tasks, LogUtils.get_stat_from_dict(avg_dict))
            if i_iter % self.progress.log_freq == 0:
                self.logger.debug(msg)
            if i_iter % self.progress.info_log_every_iteration == 0:
                self.logger.info(msg)
                avg_dict = {x: Average(x) for x in stat_keys}

            if i_iter % self.progress.valid_every_iteration == 0:
                random_seed = np.random.randint(1000)
                current_valid_stat_dict = self.eval(model=model, stage="valid", epoch=i_iter)
                current_score = current_valid_stat_dict["loss"].item()

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                }

                if current_score < current_best_score - 0.001:
                    current_best_score = current_score
                    PathUtils.save_ckp(checkpoint, dir_name=self.task_name, identifier=self.get_identifier(),
                                              index_key=self.index_key)
                    not_improve_cnt = 0
                else:
                    not_improve_cnt += 1
                    self.logger.info("OVERFITTING cnt : {}/{}, current score: {:.4f}, best score: {:.4f}".format(
                        not_improve_cnt, self.progress.max_not_impr_cnt, current_score, current_best_score))

                TorchUtils.set_random_seed(random_seed)

        return self.eval(model=model, stage="test", epoch=0)