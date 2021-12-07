from backbones.conv_net import Conv3, Conv4
from backbones.mlp_net import MLP2
from data.dataset_utils import DatasetEnum

from enum import Enum

class BackboneEnum(Enum):
    CONV4 = "conv4"
    CONV3 = "conv3"
    MLP2 = "mlp2"

DEFAULT_CLASS_NUM = 5  # 5-way classification

FEATURE_DIM_DICT = {
    # conv4
    (BackboneEnum.CONV4.name, DatasetEnum.MINI_IMAGENET.name): 1600,
    # conv3
    (BackboneEnum.CONV3.name, DatasetEnum.QMUL.name): 2916,
}


def get_embedding_dim(backbone_name, ds_name):
    if backbone_name == BackboneEnum.MLP2.name:
        return 40
    else:
        return FEATURE_DIM_DICT[(backbone_name, ds_name)]

def get_backbone(backbone_name, ds_name, config):

    embedding_dim = get_embedding_dim(backbone_name, ds_name)
    class_num = config.get("class_num", DEFAULT_CLASS_NUM)
    if backbone_name == BackboneEnum.CONV3.name:
        backbone = Conv3(embedding_dim=embedding_dim)
    elif backbone_name == BackboneEnum.CONV4.name:
        backbone = Conv4(embedding_dim=embedding_dim, out_class_num=class_num)
    elif backbone_name == BackboneEnum.MLP2.name:
        input_dim = config[ds_name]["input_dim"]
        backbone = MLP2(input_dim)
    else:
        raise ValueError("unknown backbone: {}".format(backbone_name))

    return backbone, embedding_dim
