import torch.nn.functional as F

import torch

class KernelZoo(object):

    @staticmethod
    def compute_linear(x1, x2, is_batch=True):
        if is_batch:
            assert x1.size(0) == x2.size(0)
            return x1.bmm(x2.transpose(1, 2))
        else:
            return x1.matmul(x2.t())

    @staticmethod
    def compute_rbf(x1, x2, sigma=5, is_batch=True):
        if is_batch:
            return torch.exp(-torch.norm(x1.unsqueeze(2) - x2.unsqueeze(1), dim=3, p=2) ** 2 / (2 * sigma**2))
        else:
            return torch.exp(-torch.norm(x1.unsqueeze(1) - x2, dim=2, p=2) ** 2/ (2 * sigma**2))

    @staticmethod
    def compute_cosine(x1, x2, is_batch=True):
        if is_batch:
            raise ValueError("cosine kernel hasn't yet support batch.")
        else:
            return F.cosine_similarity(x1.unsqueeze(2), x2.t().unsqueeze(0))