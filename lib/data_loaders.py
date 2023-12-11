"""
Defines a PyTorch Dataset wrapper and functions for loading CSV files generated by the :py:mod:`prepare_data` script.

Classes:
    - :class:`NormalizedImages`
    - :class:`MyYamlLoader`
    
Functions:
    - :py:meth:`get_data_transform`
    - :py:meth:`load_data_split`
    - :py:meth:`load_face_list`
    - :py:meth:`construct_include`
"""

import yaml
import os
import numpy as np
import torch.utils.data as data
from typing import Any, IO, List, Callable
from PIL import Image
import albumentations
import sys

def get_data_transform(trn_val_test_set: str,
                       config: dict) -> Callable:
    """
    Loads Albumentations <https://albumentations.ai/> transformations as a callable function.
    The callable is then used for:
        - Data augmentation (for training)
        - Normalization
        - Conversion to PyTorch Tensor 

    Args:
        trn_val_test_set (str): String specifying the preprocessing strategy from the configuration file. The typical values are either 'trn' or 'val'. The 'val' preprocessing is then also used for the 'test' split, as for neither of these splits are augmentations used.
        config (dict): Parsed configuration file.

    Returns:
        Callable: Albumentations transformation which receives a PIL Image and returns a normalized PyTorch tensor. Possibly, data augmentations are also applied.
    """
    input_size = config["model"]["input_size"]

    transform = config["preprocess"][trn_val_test_set]

    if os.path.exists(transform["path"]):
        pipeline = albumentations.load(transform["path"])
        return lambda x: pipeline(image=np.array(x))["image"]
    else:
        sys.exit(
            f"Could not find Albumentations configuration file for {transform}")


class MyYamlLoader(yaml.SafeLoader):
    """
    YAML Loader with additional `!include` constructor, allowing for including contents of other files.

    This is used for defining configuration files. Each configuration then does not need to enumerate the desired label space, but can instead use:
        'labels: !include labels0-101.yaml'
    """

    def __init__(self, stream: IO):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: MyYamlLoader, node: yaml.Node) -> Any:
    """Include file referenced at node. See :class:`.MyYamlLoader`"""

    filename = os.path.abspath(os.path.join(
        loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, MyYamlLoader)
        elif extension in ('json', ):
            return yaml.load(f)
        else:
            return ''.join(f.readlines())


yaml.add_constructor('!include', construct_include, MyYamlLoader)


def load_data_split(csv_file: str, config: dict) -> dict:
    """
    Reads a data split CSV file created by the :py:mod:`prepare_data` script.
    The data split CSV defines a single data split. 
    In other words, it defines what data (images) should be used for training, validation and testing.
    For N-fold cross-validation, N distinct data split files would therefore be used.

    The data split CSV contains the following information for each sample:
        - unique ID of the sample, used to identify the sample within `face_list.csv`, see :py:meth:`load_face_list`
        - path to the sample, pre-processed by :py:mod:`prepare_data` script (aligned, cropped, resized)
        - assignment to train (0) / val (1) / test (2), here refered to as a 'folder'. Not to be confused with 'folder' defined within `face_list.csv`, which can generally contain any numbers, not just 0,1,2
        - normalized labels (the project supports multi-head tasks, hence, more labels than the age estimation one can be provided)

    This function extracts the unique image IDs, their folders (train/test/val) and (normalized) labels.

    Args:
        csv_file (str): Path to a data_splitX.csv file to be loaded.
        config (dict): Parsed configuration file.

    Returns:
        dictionary: A dictionary of the following:
            - face_id (int array): Unique ID of the normalized face image.
            - folder (int array): Value {0,1,2} assigning the image to train, val or test part. 
            - true_label (dict): Normalized labels for prediction tasks defined in the config.
    """

    tasks = [task['tag'] for task in config['heads']]

    face_id = []
    folder = []
    true_label = {task: [] for task in tasks}
    with open(csv_file, 'r') as rf:
        for line in rf.readlines():
            record = line.split(',')
            face_id.append(record[0])
            folder.append(record[2])
            for i, task in enumerate(tasks):
                true_label[task].append(record[3+i])

    for i, task in enumerate(tasks):
        true_label[task] = np.array(true_label[task], dtype=int)

    data_split = {'face_id': np.array(face_id, dtype=int),
                  'true_label': true_label,
                  'folder': np.array(folder, dtype=int)}

    return data_split


def load_face_list(csv_file: dict) -> dict:
    """
    Reads a face list CSV file created by the :py:mod:`prepare_data` script.
    The face list CSV includes information about all samples for an experiment and is used to create data split CSV files later on, see :py:meth:`load_data_split`. 
    The CSV file links the normalized images (processed by :py:mod:`prepare_data`) with the original raw data.

    The face list CSV contains the following information for each sample:
        - unique ID of the sample, used to identify the sample
        - path to the sample, pre-processed by :py:mod:`prepare_data` script (aligned, cropped, resized)
        - ID of the database (dataset) from which the sample originates, e.g., AgeDB might have ID 0 and MORPH might have ID 1
        - ID of the sample within the database (dataset) from which it originates; this ID is unique within a single dataset, but does not have to be unique within the face list CSV
        - assignment of the sample to a folder; These folders are used to separate the data into training, validation and testing parts. For example, one might have folders 0,1,2,3,4,5 and decide to use folders 0,1,2,3 for training, folder 4 for validation and folder 5 for testing.
        - normalized labels (the project supports multi-head tasks, hence, more labels than the age estimation one can be provided)    

    This function extracts the unique image IDs, the database (dataset) IDs, the ID of the sample within the dataset and the folder.

    Args:
        csv_file (str): Path to a data_splitX.csv file to be loaded.

    Returns:
        dictionary: A dictionary of the following:
            - face_id (int array): Unique ID of the normalized face image.
            - db_id (int): ID of the original database (dataset).
            - item_id (int): ID of the sample within the original database.
            - folder (int array): Value assigning the image to a 'folder' (not the same as train/val/test part). The folders can be used to separate the data efficiently into train/val/test parts.
    """

    face_id = []
    db_id = []
    item_id = []
    folder = []
    with open(csv_file, 'r') as rf:
        for line in rf.readlines():
            record = line.split(',')
            face_id.append(record[0])
            db_id.append(record[2])
            item_id.append(record[3])
            folder.append(record[4])

    face_list = {'face_id': np.array(face_id, dtype=int),
                 'db_id': np.array(db_id, dtype=int),
                 'item_id': np.array(item_id, dtype=int),
                 'folder': np.array(folder, dtype=int)}

    return face_list


class NormalizedImages(data.Dataset):
    """
    PyTorch Dataset, constructed from a data split CSV file.

    The data split CSV defines a single data split. 
    In other words, it defines what data (images) should be used for training, validation and testing.
    For N-fold cross-validation, N distinct data split files would therefore be used.

    The data split CSV contains the following information for each sample:
        - unique ID of the sample, used to identify the sample within `face_list.csv`, see :py:meth:`load_face_list`
        - path to the sample, pre-processed by :py:mod:`prepare_data` script (aligned, cropped, resized)
        - assignment to train (0) / val (1) / test (2), here refered to as a 'folder'. Not to be confused with 'folder' defined within `face_list.csv`, which can generally contain any numbers, not just 0,1,2
        - normalized labels (the project supports multi-head tasks, hence, more labels than the age estimation one can be provided)

    This class reads the CSV file, extracts paths to the data and the corresponding labels. 
    Only specified data (i.e., train, val or test) are extracted. The Dataset then allows sampling of the loaded data.
    """

    def __init__(self, csv_file: str,
                 label_tags: List[str],
                 folders: List[int],
                 transform: Callable,
                 load_to_memory: bool = True) -> object:
        """
        Initializes the Dataset object, possibly reading the data into memory.

        Args:
            csv_file (str): Path to a data split CSV file.
            label_tags (List[str]): List of tags (attribute names) assigned by the configuration file to the different prediction heads. Typically, only a single tag 'age' is used.
            folders (List[int]): List of integers (0: train, 1: validation, 2: testing) to load. Typically, only one of the three values is used, e.g., to build a training dataset. Defaults to None.
            transform (Callable): Transformation applied on PIL Images, see :py:meth:`get_data_transform`. Defaults to None.
            load_to_memory (bool, optional): If True, pre-loads images to memory. Defaults to True.
        """
        data = self.load_csv(csv_file, folders)

        self.transform = transform
        self.num_labels = len(label_tags)
        self.label_tags = label_tags
        self.labels = {x: [] for x in label_tags}
        self.id = []
        self.images = []
        self.folder = []
        self.num_records = len(data)
        self.load_to_memory = load_to_memory

        for record in data:
            id = record[0]
            img_path = record[1]
            folder = record[2]
            if self.load_to_memory:
                img = Image.open(img_path).convert('RGB')
                self.images.append(img)
            else:
                self.images.append(img_path)
            self.id.append(id)
            self.folder.append(folder)
            for i, tag in enumerate(self.label_tags):
                self.labels[tag].append(record[3+i])

    def __getitem__(self, index: int):
        """
        Loads and returns a sample from the dataset at the given index.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple of the following:
                - PIL Image
                - Dictionary of tag: label pairs (typically, only the tag 'age' is used)
                - Unique ID identifying the sample within a face list CSV.
                - Either 0, 1, or 2, denoting training, validation or test sample.
        """
        if self.load_to_memory:
            img = self.images[index]
        else:
            img = Image.open(self.images[index]).convert('RGB')
        labels = {tag: self.labels[tag][index] for tag in self.label_tags}
        id = self.id[index]
        folder = self.folder[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, labels, id, folder

    def __len__(self):
        """
        Returns total number of samples.
        """
        return self.num_records

    def load_csv(self, csv_file: str, folders: List[int]):
        """
        Loads the data split CSV file, similarly to the :py:meth:`load_data_split` function.
        It reads the following:
            - Unique ID of the sample, used to identify the sample within `face_list.csv`, see :py:meth:`load_face_list`
            - Path to the sample, pre-processed by :py:mod:`prepare_data` script (aligned, cropped, resized)
            - Assignment to train (0) / val (1) / test (2), here refered to as a 'folder'. Not to be confused with 'folder' defined within `face_list.csv`, which can generally contain any numbers, not just 0,1,2
            - Normalized labels (the project supports multi-head tasks, hence, more labels than the age estimation one can be provided)

        The function is called during initialization of the Dataset.
        """
        data = []
        with open(csv_file, 'r') as rf:
            for line in rf.readlines():
                record = line.split(',')
                record = [int(record[0]), record[1].strip('"')] + \
                    [int(record[i]) for i in range(2, len(record))]
                folder = record[2]
                if folder in folders:
                    data.append(record)
        return data
