from data.dataset_utils import DatasetEnum
from utils.path_utils import PathUtils

import os.path as path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class QMUL_DataSet(object):
    """
    refer to https://github.com/BayesWatch/deep-kernel-transfer
    """
    def __init__(self, is_out_range=False):
        self.ds_name = DatasetEnum.QMUL.name
        self.is_out_range = is_out_range
        self.dataset_path = PathUtils.get_dataset_path(self.ds_name)
        self.dataset_img_path = path.join(self.dataset_path, "images")
        self.train_people = ['DennisPNoGlassesGrey', 'JohnGrey', 'SimonBGrey', 'SeanGGrey', 'DanJGrey', 'AdamBGrey',
                        'JackGrey', 'RichardHGrey', 'YongminYGrey', 'TomKGrey', 'PaulVGrey', 'DennisPGrey',
                        'CarlaBGrey', 'JamieSGrey', 'KateSGrey', 'DerekCGrey', 'KatherineWGrey', 'ColinPGrey',
                        'SueWGrey', 'GrahamWGrey', 'KrystynaNGrey', 'SeanGNoGlassesGrey', 'KeithCGrey', 'HeatherLGrey']
        self.test_people = ['RichardBGrey', 'TasosHGrey', 'SarahLGrey', 'AndreeaVGrey', 'YogeshRGrey']
        self.peoples = self.train_people + self.test_people
        self.train_task_ids = range(len(self.train_people))
        self.test_task_ids = range(len(self.test_people))
        self.angles = [x * 10 for x in range(19)]

    def sample_a_task(self, stage):
        task_ids = self.train_task_ids if stage == "train" else self.test_task_ids
        task_id = np.random.choice(task_ids)
        return task_id

    def sample_examples(self, task_id, num_samples, stage="train"):
        is_train = stage == "train"
        amp = np.random.uniform(-3, 3)
        phase = np.random.uniform(-5, 5)
        wave = [(amp * np.sin(phase + x)) for x in range(num_samples)]

        angles = np.random.choice(self.angles, num_samples, replace=False)
        if self.is_out_range and is_train:
            angles = np.random.choice(self.angles[0:10], num_samples, replace=True)
        pitches = [int(round(((y + 3) * 10) + 60, -1)) for y in wave]
        curve = [(p, a) for p, a in zip(pitches, angles)]

        person = self.train_people[task_id] if is_train else self.test_people[task_id]
        inps, targs = self.get_person_at_curve(person, curve)

        return inps, targs

    @staticmethod
    def num_to_str(num):
        if num == 0:
            return '000'
        elif num < 100:
            return '0' + str(int(num))
        else:
            return str(int(num))

    def get_person_at_curve(self, person, curve):
        faces = []
        targets = []

        train_transforms = transforms.Compose([transforms.ToTensor()])
        for pitch, angle in curve:
            image_name = "{}_{}_{}.jpg".format(person[:-4], QMUL_DataSet.num_to_str(pitch), QMUL_DataSet.num_to_str(angle))
            fname = path.join(self.dataset_img_path, person, image_name)
            img = Image.open(fname).convert('RGB')
            img = train_transforms(img)

            faces.append(img)
            pitch_norm = 2 * ((pitch - 60) / (120 - 60)) - 1
            targets.append(torch.Tensor([pitch_norm]))

        faces = torch.stack(faces)
        targets = torch.stack(targets).squeeze()
        return faces.cuda(), targets.cuda()

    def __str__(self):
        info = "#task train: {}\t test:{}".format(len(self.train_people), len(self.test_people))
        return "{}\t{}".format(self.ds_name, info)