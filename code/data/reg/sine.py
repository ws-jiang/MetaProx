import torch
import numpy as np

from data.dataset_utils import DatasetEnum
from utils.config_utils import ConfigUtils


class SineTask(object):
    def __init__(self, amp, phase, min_x, max_x):
        super(SineTask, self).__init__()
        self.phase = phase
        self.max_x = max_x
        self.min_x = min_x
        self.amp = amp

    def sample_data(self, num_samples=5):
        x = torch.distributions.Uniform(self.min_x, self.max_x).sample((num_samples, 1))
        y = self.true_sine(x).squeeze(axis=-1)
        return x, y

    def true_sine(self, x):
        return self.amp * torch.sin(self.phase + x)


class Sine_DataSet(object):
    def __init__(self):
        self.ds_name = DatasetEnum.SINE.name
        sine_ds_config = ConfigUtils.get_config_dict("dataset_configs.yaml")["sine_ds"]
        self.min_amp = sine_ds_config["min_amp"]
        self.max_phase = sine_ds_config["max_phase"]
        self.min_phase = sine_ds_config["min_phase"]
        self.max_amp = sine_ds_config["max_amp"]
        self.min_x = sine_ds_config["min_x"]
        self.max_x = sine_ds_config["max_x"]

        self.tasks_cnt = 11000
        self.tasks_id = list(range(self.tasks_cnt))
        self.tasks = [self.generate_a_task() for i in self.tasks_id]

        # split meta-train, meta-valid, meta-test
        np.random.shuffle(self.tasks_id)
        training_tasks_cnt = int(8/11 * self.tasks_cnt)
        valid_tasks_cnt = int(1/11 * self.tasks_cnt)
        test_tasks_cnt = self.tasks_cnt - training_tasks_cnt - valid_tasks_cnt
        self.stage_task_ids = {
                "train": self.tasks_id[0: training_tasks_cnt],
                "valid": self.tasks_id[training_tasks_cnt: training_tasks_cnt + valid_tasks_cnt],
                "test": self.tasks_id[training_tasks_cnt + valid_tasks_cnt:]
            }

    def generate_a_task(self):
        amp = np.random.uniform(self.min_amp, self.max_amp)
        phase = np.random.uniform(self.min_phase, self.max_phase)
        task = SineTask(amp, phase, self.min_x, self.max_x)
        return task

    def sample_a_task(self, stage):
        task_id = np.random.choice(self.stage_task_ids[stage])
        return task_id

    def sample_examples(self, task_id, num_samples, stage=None):
        """
        sample num examples from the task
        :param task_id:
        :param num:
        :return:
        """
        task = self.tasks[task_id]
        x, y = task.sample_data(num_samples)
        return x, y

    def __str__(self):
        info = "\t".join(["{}:{}".format(k, len(v)) for (k, v) in self.stage_task_ids.items()])
        return "{}\t #tasks: {}".format(self.ds_name, info)