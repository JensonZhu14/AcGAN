import numpy as np
import pandas as pd
from PIL import Image
import torch
from models.utils import ops
from sklearn.model_selection import train_test_split
import torch.utils.data as tordata
from torchvision import transforms
import os.path as osp
import os
import random
from torchvision.datasets.folder import pil_loader

class AgeDataset(tordata.Dataset):
    def __init__(self, test_size, dataset_name, age_group, data_root, split='train', transforms=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.age_group = age_group
        self.split = split
        self.test_size = test_size
        self.transforms = transforms
        # get images
        image_names = np.array(os.listdir(data_root))
        image_labels = np.array(
            [ops.age_to_group(ops.get_age_label(x, self.dataset_name), self.age_group) for x in image_names]).astype(
            int)
        train_image_names, test_image_names, train_image_labels, test_image_labels = train_test_split(image_names,
                                                                                                      image_labels,
                                                                                                      test_size=test_size,
                                                                                                      random_state=0)
        if split == 'train':
            self.image_list = [os.path.join(data_root, x) for x in train_image_names]
            self.labels = train_image_labels
        else:
            self.image_list = [os.path.join(data_root, x) for x in test_image_names]
            self.labels = test_image_labels

        print('Load {} dataset: {}, # of images: {}'.format(self.split, self.dataset_name, len(self.image_list)))

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        age = self.labels[index]

        return img, age

    def __len__(self):
        return len(self.image_list)
