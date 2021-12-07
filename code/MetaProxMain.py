import copy
import torch
import yaml
import ast
from backbones.backbone_utils import BackboneEnum
from baselines.metaprox_reg import MetaProxRegression
from baselines.metaprox_cvxlayer import MetaProxCVXLayer
from baselines.method_enum import MethodEnum
from data.dataset_utils import DatasetEnum
from utils.config_utils import ConfigUtils
from utils.path_utils import PathUtils
from utils.progress_utils import ExptEnum
from utils.time_utils import TimeUtils
from utils.torch_utils import TorchUtils

# set working path
from pathlib import Path
work_path = Path().resolve().parent
PathUtils.HOME_PATH = work_path
PathUtils.set_path()
print(work_path)

def get_method(method_name):
    if method_name == MethodEnum.MetaProx.name:
        return MetaProxCVXLayer

def regression_main(args):
    index_key = TimeUtils.get_now_str(fmt=TimeUtils.YYYYMMDDHHMMSS_COMPACT)
    config_dict_template = ConfigUtils.get_basic_config()
    config_file = "meta_prox_regression.yaml"

    config_dict_template.update(ConfigUtils.get_config_dict(config_file))
    config_dict = copy.deepcopy(config_dict_template)

    TorchUtils.set_random_seed(seed)
    config_dict["backbone_name"] = args.backbone
    config_dict["noise_sigma"] = args.noise_sigma
    config_dict["index_key"] = index_key
    config_dict["is_out_range"] = args.is_out_range
    config_dict["n_support"] = config_dict[args.expt]["n_support"]
    config_dict["n_query"] = config_dict[args.expt]["n_query"]
    config_dict["ds"] = args.ds
    job_id = "{}_{}_{}_{}_{}".format(config_dict["ds"], "MetaProx", config_dict["n_support"], int(args.noise_sigma), config_dict["is_out_range"])
    config_dict["job_id"] = job_id

    meta_prox = MetaProxRegression(config_dict)
    meta_prox.logger.info(yaml.dump(config_dict, default_flow_style=False))
    meta_prox.train(seed=args.seed)

def classification_main(args):
    index_key = TimeUtils.get_now_str(fmt=TimeUtils.YYYYMMDDHHMMSS_COMPACT)

    config_dict_template = ConfigUtils.get_basic_config()
    config_file = "baseline_cls.yaml"
    config_dict_template.update(ConfigUtils.get_config_dict(config_file))

    config_file = "meta_prox_classification.yaml"
    config_dict_template.update(ConfigUtils.get_config_dict(config_file))

    seed = args.seed
    TorchUtils.set_random_seed(seed)

    config_dict = copy.deepcopy(config_dict_template)

    config_dict["backbone_name"] = args.backbone
    config_dict["experiment"] = args.expt
    config_dict["index_key"] = index_key
    config_dict["method_name"] = args.method
    config_dict["ds_name"] = args.ds
    job_id = "{}_{}_{}_{}_{}".format(args.method, args.ds, args.expt, args.backbone, args.job_type)
    config_dict["job_id"] = job_id

    method_class = get_method(method_name=args.method)
    method = method_class(config_dict)
    method.logger.info(config_dict)
    method.train(seed=seed)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## default
    gpu_id = 0
    ds = DatasetEnum.MINI_IMAGENET.name
    expt = ExptEnum.S_5_WAY_5_SHOT.value
    method = MethodEnum.MetaProx.name
    backbone = BackboneEnum.CONV4.name
    seed = 101
    job_type = "release_{}".format(seed)
    seeds = [seed]

    ##
    parser.add_argument('--gpu_id', default=gpu_id, type=int)
    parser.add_argument('--job_type', default=job_type, type=str)
    parser.add_argument('--ds', default=ds, type=str)
    parser.add_argument('--expt', default=expt, type=str)
    parser.add_argument('--method', default=method, type=str)
    parser.add_argument('--backbone', default=backbone, type=str)
    parser.add_argument('--cls_or_reg', default="cls", type=str)
    parser.add_argument('--noise_sigma', default=0, type=int) # for reg
    parser.add_argument('--is_out_range', type=ast.literal_eval, default=False) # for QMUL
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--seeds', default=seeds, nargs='+', type=int)
    args = parser.parse_args()
    print(args)

    device_id = args.gpu_id
    torch.cuda.set_device(device=device_id)
    if args.cls_or_reg == "cls":
        classification_main(args)
    else:
        regression_main(args)