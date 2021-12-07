
import numpy as np
import torch


class NWayKShotSampler:

    def __init__(self, labels, total_steps=20, n_way=5, k_shot=1, start_fraction=0, end_fraction=1, is_shuffle_head=True):
        """
        n way k shot sampler for few-shot learning problem
        :param labels: list, index means sample index, value means label
        :param total_steps: train step number
        :param n_way: number of class per batch
        :param k_shot: number of samples per class in one batch
        """
        self.total_steps = total_steps
        self.n_way = n_way
        self.k_shot = k_shot

        self.label_2_instance_ids = []

        for i in range(max(labels) + 1):
            ids = np.argwhere(np.array(labels) == i).reshape(-1)
            ids = torch.from_numpy(ids)
            num_instance = len(ids)
            start_id = max(0, int(np.floor(start_fraction * num_instance)))
            end_id = min(num_instance, int(np.floor(end_fraction * num_instance)))
            self.label_2_instance_ids.append(ids[start_id: end_id])

        self.labels_num = len(self.label_2_instance_ids)
        self.labels = labels

        self.block_size = int(self.labels_num / self.n_way)

        self.half_size = int(self.labels_num / 2)
        self.is_shuffle_head = is_shuffle_head

        self.class_ids_shuffle = torch.randperm(self.labels_num)

    def __len__(self):
        return self.total_steps

    def __iter__(self):

        for i_batch in range(self.total_steps):
            batch = []
            class_ids = []
            if self.is_shuffle_head:
                class_ids = torch.randperm(self.labels_num)[0:self.n_way]  # pick n_class randomly
            else:
                for i in range(self.n_way):
                    rand_int = torch.randperm(self.block_size)[0]
                    class_ids.append(self.block_size * i + rand_int)

            for class_id in class_ids:
                instances_ids = self.label_2_instance_ids[class_id]
                # pick n_instances per each class randomly
                instances_ids_selected = torch.randperm(len(instances_ids))[0:self.k_shot]
                batch.append(instances_ids[instances_ids_selected])
            batch = torch.stack(batch).reshape(-1)
            yield batch

