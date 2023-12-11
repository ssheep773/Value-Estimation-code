import torch
import numpy as np
import os
import pandas as pd 
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from torch.utils.data import Dataset
try:
    from timm.data.transforms import str_to_pil_interp as _pil_interp
except:
    from timm.data.transforms import _pil_interp

from PIL import Image
import csv

class MyDataset(Dataset):
    def __init__(self, organized_data, transform):
        self.organized_data = organized_data
        self.transform = transform    # 影像的轉換方式

    def __getitem__(self, idx):
        entry = self.organized_data[idx].copy()
        filename_value = entry.pop('filename', None)
        image = Image.open(filename_value).convert('RGB')
        label = entry

        if self.transform is not None:
          image = self.transform(image) # Transform image

        return image, label           # return 模型訓練所需的資訊

    def __len__(self):
        return len(self.organized_data)    # return DataSet 長度

def build_loader(config):
    dataset_train = build_dataset(is_train=True, config=config)
    dataset_val = build_dataset(is_train=False, config=config)

    sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=False)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config["optimizer"]["batch_size"],
        num_workers=config["optimizer"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config["optimizer"]["batch_size"],
        shuffle=False,
        num_workers=config["optimizer"]["num_workers"],
        pin_memory=True,
        drop_last=False
    )

    return data_loader_train, data_loader_val

def build_dataset(is_train, config, test=False):
    transform = build_transform(is_train, config)
    
    if test:
        label_csv = os.path.join(config["data"]["data_dir"], "test_feature_groundturth.csv")
        data_dir = os.path.join(config["data"]["data_dir"], "test", "feature")
    else:
        if is_train:
            label_csv = os.path.join(config["data"]["data_dir"], "train_feature_groundturth.csv")
            data_dir = os.path.join(config["data"]["data_dir"], "train", "feature")
        else:
            label_csv = os.path.join(config["data"]["data_dir"], "valid_feature_groundturth.csv")
            data_dir = os.path.join(config["data"]["data_dir"], "valid", "feature")

    with open(label_csv, "r") as csv_file:
        organized_data = []
        csv_reader = csv.DictReader(csv_file)
        
        for entry in csv_reader:
            entry['filename'] = os.path.join(data_dir, entry['filename'])
            entry['NID'] = float(entry['NID'])
            entry['S'] = float(entry['S'])
            entry['L'] = float(entry['L'])
            entry['R'] = float(entry['R'])

            organized_data.append(entry)

    dataset = MyDataset(organized_data, transform)
    return dataset

def build_transform(is_train, config):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(256, 256), scale=(1.0, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        
    return transform
