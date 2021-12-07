from data.cls.miniimagenet import MiniImagenet_DataSet
from data.dataset_utils import DatasetEnum
from data.reg.qmul import QMUL_DataSet
from data.reg.sale import Sale_DataSet
from data.reg.sine import Sine_DataSet

def get_dataset(ds_name, split, is_out_range=False):
    # cls
    if ds_name == DatasetEnum.MINI_IMAGENET.name:
        return MiniImagenet_DataSet(split=split)
    # reg
    elif ds_name == DatasetEnum.SALE.name:
        return Sale_DataSet()
    elif ds_name == DatasetEnum.SINE.name:
        return Sine_DataSet()
    elif ds_name == DatasetEnum.QMUL.name:
        return QMUL_DataSet(is_out_range=is_out_range)
    else:
        raise ValueError("unknown dataset: {}, {}".format(ds_name, split))
