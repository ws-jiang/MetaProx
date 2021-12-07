from utils.path_utils import PathUtils

import os
import logging
from pathlib import Path


class LogUtils:

    LOGGER_FORMATTER = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    LOGGER_DICT = {}

    def __init__(self):
        pass

    @staticmethod
    def print_files_in_dir(temp_dir):
        """
        print all the files in this directory
        :param temp_dir:
        :return:
        """
        x = os.listdir(temp_dir)
        for file in x:
            print(file)

    @staticmethod
    def get_dict_info(config_dict):
        import json
        return json.dumps(config_dict,  sort_keys=True, indent=4)

    @staticmethod
    def get_or_init_logger(file_name, dir_name=None, level=logging.DEBUG) -> logging.Logger:
        """To setup as many loggers as you want"""
        if dir_name is None:
            raise ValueError("job id should not be None!")

        log_dir_path = "{}/{}".format(PathUtils.Log_HOME_PATH, dir_name)
        log_file = "{}/log_{}.log".format(log_dir_path, file_name)


        if log_file in LogUtils.LOGGER_DICT:
            return LogUtils.LOGGER_DICT[log_file]

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # return logger if exists
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(LogUtils.LOGGER_FORMATTER)
        logger = logging.getLogger(file_name)
        logger.setLevel(level)
        logger.addHandler(handler)

        LogUtils.LOGGER_DICT[log_file] = logger
        os.utime(Path(log_dir_path))

        return logger

    @staticmethod
    def get_stat_from_dict(stat_dict, keys=None):
        if not keys:
            keys = stat_dict.keys()

        msg = []
        for key in keys:
            msg.append(str(stat_dict[key]))

        return "; ".join(msg)

