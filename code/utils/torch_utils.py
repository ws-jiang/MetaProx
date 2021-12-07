
from torch import nn
import torch
import numpy as np

class TorchUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def set_random_seed(seed):
        seed_id = seed
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def copy_model_parameters(module_src, module_dest):
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data.copy_(param.data)

    @staticmethod
    def update_model_parameters(module_src, module_dest, lr=0.001):
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data.copy_(lr * param.data + (1-lr)*dict_dest[name].data)

    @staticmethod
    def init_weights(module):
        for m in module.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def set_parameter_requires_grad(model, enable=True):
        for param in model.parameters():
            param.requires_grad = enable

def accumulate(inputs):
    """
    inputs = [1, 2, 3, 4]
    return = [1, 3, 6, 10]
    :param inputs:
    :return:
    """
    total = 0
    result = []
    for x in inputs:
        total += x
        result.append(total)
    return result