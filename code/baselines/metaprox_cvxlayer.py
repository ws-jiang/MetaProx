import cvxpy as cp
import torch
import torch.nn.functional as F
from cvxpy import MatrixFrac
from cvxpylayers.torch import CvxpyLayer
from torch import nn

from backbones.backbone_utils import get_backbone
from baselines.abstract_alg import AbstractAlg
from baselines.kernels import KernelZoo as K
from utils.tensor_utils import split_support_query_for_x_in_cls, one_hot_embedding


class MetaProxModel(nn.Module):
    def __init__(self, config):
        super(MetaProxModel, self).__init__()
        self.config = config
        self.model, _ = get_backbone(backbone_name=config["backbone_name"], ds_name=config["ds_name"], config=self.config)
        self.alpha = 10

        # theta
        self.n_way = 5
        self.proximal_model = nn.Linear(self.n_way, self.n_way, bias=False)
        with torch.no_grad():
            self.proximal_model.weight = torch.nn.Parameter(torch.eye(self.n_way).cuda().float())

    def forward_feature(self, x):
        return self.model.forward_feature(x)


class MetaProxCVXLayer(AbstractAlg):
    def __init__(self, config):
        super(MetaProxCVXLayer, self).__init__(config=config)


    def task_update(self, x_support, y_support, n_way, stage, model):
        """
        task update aims to solve the dual problem in the inner loop
        using cvxpylayer: https://locuslab.github.io/2019-10-28-cvxpylayers/
        Note: for some CPUs, `cvxlayer` may be unstable and fail to solve the cvx opt problem.
        x_support: n_support * d
        y_support: n_support * n_class
        """
        n_support = x_support.size(0)
        kernel_matrix = K.compute_cosine(x_support, x_support, is_batch=False)  # n_support x n_support

        label_en_matrix = one_hot_embedding(y_support, n_way)
        label_en_matrix = label_en_matrix.float().cuda()

        G = kernel_matrix
        G = torch.inverse(G)
        protos = x_support.contiguous().view(n_way, self.n_support_dict[stage], -1).mean(dim=1)
        D = K.compute_cosine(x_support, protos, is_batch=False).matmul(model.proximal_model.weight)

        alpha = cp.Variable((n_support, n_way))
        constraints = [0 <= label_en_matrix.cpu().numpy() - alpha, cp.sum(alpha, axis=1) == 0]

        G_para = cp.Parameter(shape=(n_support, n_support), PSD=True)
        D_para = cp.Parameter((n_support, n_way))

        loss = - cp.sum(cp.entr(label_en_matrix.cpu().numpy() - alpha)) + cp.sum(
            cp.multiply(D_para, alpha)) + MatrixFrac(alpha, G_para)
        objective = cp.Minimize(loss)

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        try:
            cvxpylayer = CvxpyLayer(problem, parameters=[G_para, D_para], variables=[alpha])
            alpha_result = cvxpylayer(G, D)
            return alpha_result[0], True
        except Exception as e:
            self.logger.warn("unable to solve cvx in base learner")
            return None, False

    def meta_update(self, x, stage):
        n_way, n_support, n_query, y_support, y_query = self.get_basic_expt_info(stage)
        x = self.model.forward_feature(x)
        x = x - x.mean(0)
        x = x.contiguous().view(n_way, n_support + n_query, -1)
        x_support, x_query = split_support_query_for_x_in_cls(x, n_support=n_support)
        x_support = x_support.contiguous().view(n_way*n_support, -1)
        x_query = x_query.contiguous().view(n_way*n_query, -1)

        protos = x_support.contiguous().view(n_way, self.n_support_dict[stage], -1).mean(dim=1)  # n_way * feature_dim
        alpha, task_update_status = self.task_update(x_support, y_support, n_way, stage, self.model)

        y_pred_prior = K.compute_cosine(x_query, protos, is_batch=False).matmul(self.model.proximal_model.weight)
        if task_update_status:
            y_pred_posterior = K.compute_cosine(x_query, x_support, is_batch=False).matmul(alpha)
            y_query_pred = y_pred_prior + y_pred_posterior
        else:
            y_query_pred = y_pred_prior

        query_loss = F.cross_entropy(self.model.alpha * y_query_pred, y_query)

        result_dict = {
            "y_query_pred": y_query_pred,
            "query_loss": query_loss,
            "meta_loss": query_loss,
        }

        return result_dict

    def get_model(self):
        return MetaProxModel(config=self.config)