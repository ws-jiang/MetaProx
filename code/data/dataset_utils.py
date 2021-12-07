from enum import Enum

class DatasetEnum(Enum):
    # name: ds name, value: ds path
    ## classification
    MINI_IMAGENET = "mini-imagenet"

    ## regression
    QMUL = "QMUL"
    SINE = "sine"
    SALE = "sales"

    @classmethod
    def get_value_by_name(cls, ds_name):
        ds_dict = dict([(ds.name, ds.value) for ds in DatasetEnum])
        return ds_dict[ds_name]