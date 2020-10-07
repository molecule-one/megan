# -*- coding: utf-8 -*-
"""
Base class for a dataset
"""


import os
from abc import ABCMeta, abstractmethod

import pandas as pd

from src import DATA_DIR


class Dataset(metaclass=ABCMeta):
    """
    Base class for all datasets. Dataset can be downloaded or generated, but must have a similar final format.
    After preprocessing, all datasets should consists of a two files: x.tsv, metadata.tsv
    x.tsv contains columns 'substrates' and 'product' (SMILES)
    metadata.tsv contain contain additional information such as reaction type, class etc.
    """
    def __init__(self):
        super(Dataset, self).__init__()
        self._create_directories()

    def _create_directories(self):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        if not os.path.exists(self.feat_dir):
            os.mkdir(self.feat_dir)

    @property
    def meta_info(self) -> dict:
        """
        :return: some hard-coded meta information about the dataset
        """
        return {}

    @property
    @abstractmethod
    def key(self) -> str:
        """
        :return: key of the dataset as a string. Defines dataset directory.
        """
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def acquire(self):
        """
        Acquire the dataset by downloading it from some source or generate it.
        """
        raise NotImplementedError("Abstract method")

    def load_x(self) -> pd.DataFrame:
        """
        Load X of an acquired dataset. Yields exception if dataset is not acquired.
        :return: dataset as a pandas DataFrame
        """
        if not self.is_acquired():
            raise FileNotFoundError('Dataset has not been downloaded and/or generated. '
                                    'Please run "acquire" method to be able to load it!')
        return pd.read_csv(self.x_path, sep='\t')

    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata of an acquired dataset. Yields exception if dataset is not acquired.
        :return: metadata as a pandas DataFrame
        """
        if not self.is_acquired():
            raise FileNotFoundError('Dataset has not been downloaded and/or generated. '
                                    'Please run "acquire" method to be able to load it!')
        return pd.read_csv(self.metadata_path, sep='\t')

    @property
    def dir(self) -> str:
        """
        :return: directory of the dataset data, which depends on DATA_DIR env variable and the key of the dataset.
        """
        return os.path.join(DATA_DIR, self.key)

    @property
    def x_path(self) -> str:
        """
        :return: path to a .tsv containing SMILES and/or other 'X' (e. g. changed_atoms) of the processed dataset.
        """
        return os.path.join(self.dir, 'x.tsv')

    @property
    def metadata_path(self) -> str:
        """
        :return: path to a .tsv containing metadata of the processed dataset.
        """
        return os.path.join(self.dir, 'metadata.tsv')

    def is_acquired(self) -> bool:
        """
        Check whether the dataset has already been acquired (e. g. to omit unnecessary second download/generation)
        """
        return os.path.exists(self.x_path) and os.path.exists(self.metadata_path)

    @property
    def feat_dir(self) -> str:
        """
        :return: directory in which featurized versions of the dataset will be stored.
        """
        return os.path.join(self.dir, 'feat')
