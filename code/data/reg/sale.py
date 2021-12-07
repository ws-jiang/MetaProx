import os.path as path
import numpy as np
import pandas as pd
import torch

from utils.path_utils import PathUtils
from data.dataset_utils import DatasetEnum


class Sale_DataSet(object):
    """
    refer to https://arxiv.org/pdf/1811.05695.pdf and https://www.kaggle.com/crawford/weekly-sales-transactions
    https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly
    """
    def __init__(self):
        self.ds_name = DatasetEnum.SALE.name

        self.feature_size = 5

        data_home_path = PathUtils.get_dataset_path(self.ds_name)
        data_path = path.join(data_home_path, "Sales_Transactions_Dataset_Weekly.csv")
        sales_df = pd.read_csv(data_path)

        sales = sales_df[sales_df.columns[-52:]].values  # only the normalized feature

        self.tasks_cnt = sales.shape[0]

        week_num = 52
        self.samples = [] # list of sample feature
        self.labels = [] # list of sample label
        self.samples_task_index = {}  # task_id -> list of samples_id
        sample_id = 0
        for task_id_ in range(self.tasks_cnt):
            self.samples_task_index[task_id_] = []
            for t in range(self.feature_size, week_num):
                self.samples.append(sales[task_id_, t - self.feature_size: t])
                self.labels.append(sales[task_id_][t])
                self.samples_task_index[task_id_].append(sample_id)
                sample_id += 1

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        # train, valid, test = (600, 100, 111)
        training_tasks_cnt = 600
        validation_tasks_cnt = 100
        tasks_id = list(range(self.tasks_cnt))
        np.random.shuffle(tasks_id)

        self.stage_task_ids = {
            "train": tasks_id[0: training_tasks_cnt],
            "valid": tasks_id[training_tasks_cnt: training_tasks_cnt + validation_tasks_cnt],
            "test": tasks_id[training_tasks_cnt + validation_tasks_cnt:]
        }

    def sample_a_task(self, stage):
        """
        sample a task id
        :param stage:
        :return:
        """
        task_id = np.random.choice(self.stage_task_ids[stage])
        return task_id

    def sample_examples(self, task_id, num_samples, stage=None):
        """
        sample num examples from the task
        :param task_id:
        :param num:
        :return:
        """
        sample_ids = np.random.choice(self.samples_task_index[task_id], num_samples, replace=False)
        return torch.from_numpy(self.samples[sample_ids]).float(), torch.from_numpy(self.labels[sample_ids]).float()

    def __len__(self):
        return len(self.samples)

    def __str__(self):
        info = "\t".join(["{}:{}".format(k, len(v)) for (k, v) in self.stage_task_ids.items()])
        return "{}\t #tasks: {}".format(self.ds_name, info)
