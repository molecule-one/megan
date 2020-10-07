# -*- coding: utf-8 -*-
"""
Base split class
"""
import os
import pandas as pd

from abc import abstractmethod, ABCMeta

from src.datasets import Dataset

SPLITS = ['train', 'valid', 'test']


class DatasetSplit(metaclass=ABCMeta):
    """
    Base class for algorithms splitting reaction dataset.
    """

    def __init__(self):
        super(DatasetSplit, self).__init__()

    @property
    @abstractmethod
    def key(self) -> str:
        """
        :return: a string identifier of the split method
        """
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def split_dataset(self, dataset: Dataset):
        """
        Splits dataset into several clusters and saves results.
        :param dataset: dataset to split
        """
        raise NotImplementedError("Abstract method")

    def path(self, data_dir: str) -> str:
        """
        :param data_dir: base directory for the dataset
        :return: complete path to this split
        """
        return os.path.join(data_dir, f'{self.key}_split.csv')

    def load(self, data_dir: str) -> pd.DataFrame:
        """
        :param data_dir: base directory for the dataset
        :return: split as a pandas DataFrame
        """
        return pd.read_csv(self.path(data_dir))

    def has_finished(self, splits_dir: dir) -> bool:
        """
        :param splits_dir: base directory for splits for a dataset
        :return: whether the split is already computed and saved
        """
        return os.path.exists(self.path(splits_dir))

