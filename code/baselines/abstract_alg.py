from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from data.fsl_samplers import NWayKShotSampler
from data.get_dataset import get_dataset
from utils.evaluation_utils import Average
from utils.evaluation_utils import count_acc
from utils.log_utils import LogUtils
from utils.path_utils import PathUtils
from utils.progress_utils import ProgressConfig
from utils.progress_utils import StageEnum
from utils.torch_utils import TorchUtils


class AbstractAlg(ABC):
    def __init__(self, config):
        self.config = config
        self.stages = StageEnum.get_all_stages()
        self.experiment_config = config[config["experiment"]]
        self.ways = {x: self.experiment_config["way"][x] for x in self.stages}
        self.n_support_dict = self.experiment_config["shot"]["support"]
        self.n_query_dict = self.experiment_config["shot"]["query"]

        self.ds_name = self.config["ds_name"]
        self.index_key = self.config["index_key"]
        self.job_id = self.config["job_id"]
        self.method_name = self.config["method_name"]
        self.task_name = "{}_{}".format(self.method_name, "cls")
        self.logger = LogUtils.get_or_init_logger(dir_name=self.task_name, file_name=self.job_id)

        self.meta_lr = self.config.get("{}_lr".format(self.method_name), self.config["default_lr"])["meta_lr"]
        self.base_lr = self.config.get("{}_lr".format(self.method_name), self.config["default_lr"])["base_lr"]
        self.n_inner_step_dict = self.config.get("{}_inner_step".format(self.method_name), self.config["inner_step"])

        self.logger.warn("use lr: {}, {}, inner step: {}".format(self.meta_lr, self.base_lr, self.n_inner_step_dict))

        self.backbone_name = self.config["backbone_name"]
        self.progress = ProgressConfig(self.config)

        self.model = self.get_model()
        self.model.train(True)
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma=0.5, last_epoch=-1)

        self.m_dataset = {
            stage: get_dataset(ds_name=self.ds_name, split=stage)
            for stage in self.stages
        }
        self.m_sampler = {
            stage: NWayKShotSampler(self.m_dataset[stage].labels,
                                          total_steps=self.config["num_iteration"][stage],
                                          n_way=self.ways[stage], k_shot=self.n_support_dict[stage] + self.n_query_dict[stage],
                                          ) # (StageEnum.TRAIN.value == stage)
            for stage in self.stages
        }
        self.m_dataloader = {
            stage: DataLoader(self.m_dataset[stage], batch_sampler=self.m_sampler[stage],
                              num_workers=self.progress.num_workers,
                              pin_memory=True)
            for stage in self.stages
        }

        stat_keys_temp = ["loss", "acc"]
        self.stat_keys = []
        for x in stat_keys_temp:
            self.stat_keys.append(x)

        self.stat_keys.append("reg_loss")

    def get_identifier(self):
        return self.job_id

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def meta_update(self, x, stage):
        pass

    def get_best_model(self):
        return "{}_{}".format(self.config["index_key"], "best")

    def add_w(self, avg_dict, w_tensor):
        for i in range(len(w_tensor)):
            avg_dict["{}".format(i)].add(w_tensor[i].item())

    def get_basic_expt_info(self, stage):
        n_way = self.ways[stage]
        n_support = self.n_support_dict[stage]
        n_query = self.n_query_dict[stage]
        y_support = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        return n_way, n_support, n_query, y_support, y_query

    def get_w_dict(self):
        return {"{}".format(x): Average("{}".format(x)) for x in range(self.config["n_dim"])}

    def eval(self, stage=StageEnum.VALID.value, seed=0, epoch=0, is_reload=False):
        n_way, n_support, n_query, y_support, y_query = self.get_basic_expt_info(stage)
        avg_dict = {x: Average(x) for x in self.stat_keys}

        if stage == StageEnum.TEST.value and is_reload:
            TorchUtils.set_random_seed(seed)
            checkpoint = PathUtils.load_ckp(dir_name=self.task_name, identifier=self.get_identifier(),
                                            index_key=self.index_key)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.train(False)
        for i_ter, (x, _) in enumerate(self.m_dataloader[stage], start=1):
            x = x.cuda()
            result_dict = self.meta_update(x, stage=stage)
            acc = count_acc(result_dict["y_query_pred"], y_query)
            avg_dict["loss"].add(result_dict["query_loss"].item())
            avg_dict["acc"].add(acc)

            if i_ter % self.progress.log_freq == 0:
                msg = "{} {}:\t{}/{} {}\t{}".format(self.get_identifier(), stage.upper(),
                                                       i_ter, len(self.m_dataloader[stage]), epoch,
                                                       LogUtils.get_stat_from_dict(avg_dict))
                self.logger.debug(msg)

        msg = "{} {}\t{}\t{}".format(self.get_identifier(), stage.upper(), epoch, LogUtils.get_stat_from_dict(avg_dict))
        self.logger.info(msg)
        self.model.train(True)

        return avg_dict

    def train(self, num_epoch=10, seed=1):
        current_best_score = float("-inf")
        not_improve_cnt = 0

        TorchUtils.set_random_seed(seed)
        stage = StageEnum.TRAIN.value

        n_way, n_support, n_query, y_support, y_query = self.get_basic_expt_info(stage)

        num_train_iteration = self.config["num_iteration"][stage]
        avg_dict = {x: Average(x) for x in self.stat_keys}
        for i_iter, (x, _) in enumerate(self.m_dataloader[stage], start=1):
            if not_improve_cnt >= self.progress.max_not_impr_cnt:
                break

            x = x.cuda()
            result_dict = self.meta_update(x, stage=stage)
            acc = count_acc(result_dict["y_query_pred"], y_query)

            avg_dict["loss"].add(result_dict["query_loss"].item())
            avg_dict["acc"].add(acc)

            self.optimizer.zero_grad()
            result_dict["meta_loss"].backward()
            self.optimizer.step()

            if i_iter % self.progress.log_freq == 0:
                msg = "{} {}:\t{}/{}\t{}".format(self.get_identifier(), stage.upper(),
                                                       i_iter, len(self.m_dataloader[stage]),
                                                       LogUtils.get_stat_from_dict(avg_dict))
                self.logger.debug(msg)

            if i_iter % self.progress.valid_every_iteration == 0:
                current_valid_stat_dict = self.eval(seed=seed, stage=StageEnum.VALID.value, epoch=i_iter)
                current_score = current_valid_stat_dict["acc"].item()
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                }

                if current_score > current_best_score + 0.0005:
                    current_best_score = current_score
                    PathUtils.save_ckp(checkpoint, dir_name=self.task_name, identifier=self.get_identifier(),
                                       index_key=self.index_key)
                    not_improve_cnt = 0
                else:
                    not_improve_cnt += 1
                    self.logger.info("OVERFITTING cnt : {}/{}, current score: {:.4f}, best score: {:.4f}".format(
                        not_improve_cnt, self.progress.max_not_impr_cnt, current_score, current_best_score))

            if i_iter % self.progress.info_log_every_iteration == 0:
                msg = "{} {}:\t{}/{}\t{}".format(self.get_identifier(), stage.upper(), i_iter, num_train_iteration,
                                                 LogUtils.get_stat_from_dict(avg_dict))
                self.logger.info(msg)
                avg_dict = {x: Average(x) for x in self.stat_keys}
                self.lr_scheduler.step()
        self.eval(seed=seed, stage=StageEnum.TEST.value, is_reload=True)


