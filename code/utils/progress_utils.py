from enum import Enum

class StageEnum(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    @staticmethod
    def get_all_stages():
        return [StageEnum.TRAIN.value, StageEnum.VALID.value, StageEnum.TEST.value]

class MetricEnum(Enum):
    ACC = "acc"
    LOSS = "loss"

class ExptEnum(Enum):
    S_5_SHOT = "5shot"
    S_2_SHOT = "2shot"
    S_5_WAY_1_SHOT = "5way1shot"
    H_5_WAY_1_SHOT = "h5way1shot"
    S_5_WAY_5_SHOT = "5way5shot"
    H_5_WAY_5_SHOT = "h5way5shot"

class ProgressConfig(object):
    def __init__(self, config):
        self.log_freq = config.get("log_freq", 10)
        self.valid_every_iteration = config.get("valid_every_iteration", 500)
        self.info_log_every_iteration = config.get("info_log_every_iteration", 100)
        self.report_train_every_iteration = config.get("report_train_every_iteration", 1000)
        self.test_every_valid = config.get("test_every_valid", 2)
        self.max_not_impr_cnt = config.get("max_not_impr_cnt", 10)
        self.num_workers = config.get("num_workers", 16)
        self.default_lr = config.get("lr", 0.001)