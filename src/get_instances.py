"""
Functions for loading configured model, dataset and featurizer instances
"""

import gin
from src import config
from src.datasets import Dataset
from src.feat import ReactionFeaturizer
from src.split import DatasetSplit


@gin.configurable()
def get_dataset(dataset_key: str = gin.REQUIRED) -> Dataset:
    return config.get_dataset(dataset_key)


@gin.configurable()
def get_featurizer(featurizer_key: str = gin.REQUIRED) -> ReactionFeaturizer:
    return config.get_featurizer(featurizer_key)


@gin.configurable()
def get_split(split_key: str = gin.REQUIRED) -> DatasetSplit:
    return config.get_split(split_key)
