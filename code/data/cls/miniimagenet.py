
from os import path
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils.path_utils import PathUtils
from data.dataset_utils import DatasetEnum


class MiniImagenet_DataSet(Dataset):
    def __init__(self, split):
        """
        :param transform:
        :param split: train, val, test
        :param is_unit_test:
        """
        # 1. read the csv file
        # 2. load the sample
        self.ds_name = DatasetEnum.MINI_IMAGENET.name
        self.split = split

        csv_path = path.join(PathUtils.get_dataset_path(self.ds_name), "split", "{}.csv".format(split))
        images_path = path.join(PathUtils.get_dataset_path(self.ds_name), "images")

        lines = [x.strip() for x in open(csv_path).readlines()[1:]]

        self.samples = []
        self.labels = []
        label_dict = {}
        label_index = 0
        self.label_index_2_name = []
        for e in lines:
            image_name, label_name = e.split(",")
            if label_name not in label_dict:
                label_dict[label_name] = label_index
                label_index += 1
                self.label_index_2_name.append(label_name)

            self.samples.append(path.join(images_path, image_name))
            self.labels.append(label_dict[label_name])

        is_train = split == "train"
        mean, std, image_size = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844], 84
        normalization_transform = transforms.Normalize(mean=mean,  std=std)

        # transforms follows: https://github.com/kjunelee/MetaOptNet/blob/master/data/mini_imagenet.py
        self.transform = None
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalization_transform,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalization_transform,
            ])

    def __getitem__(self, index):
        image_path, _ = self.samples[index], self.labels[index]
        image = self.transform(Image.open(image_path).convert('RGB'))

        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(self.ds_name, self.split, len(self.samples), len(set(self.labels)))
