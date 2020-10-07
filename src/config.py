# -*- coding: utf-8 -*-
"""
Configuration of datasets and featurizers and splits
"""

from src import N_JOBS
from src.feat import ReactionFeaturizer
from src.datasets import Dataset
from src.datasets.uspto_50k import Uspto50k
from src.datasets.uspto_full import UsptoFull
from src.datasets.uspto_mit import UsptoMit
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer
from src.split import DatasetSplit
from src.split.basic_splits import DefaultSplit

DEFAULT_SPLIT = DefaultSplit()

# we can potentially define different kinds of splits (randomized, fingerprint-based) for each dataset
# in this work we always use the one "default" split which is randomized
SPLIT_INITIALIZERS = {
    'default': lambda **kwargs: DEFAULT_SPLIT
}

DATASET_INITIALIZERS = {
    'uspto_50k': lambda: Uspto50k(),
    'uspto_mit': lambda: UsptoMit(),
    'uspto_full': lambda: UsptoFull()
}

# all featurizer variants that we used in experiments
FEATURIZER_INITIALIZERS = {
    # 5 variants of action ordering tested on UPSTO-50k, 'megan_16_bfs_randat' is the one we use for final evaluation
    'megan_16_dfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                split=DEFAULT_SPLIT,
                                                                action_order='dfs'),
    'megan_16_bfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                split=DEFAULT_SPLIT,
                                                                action_order='bfs'),
    'megan_16_dfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT,
                                                                  action_order='dfs_randat'),
    'megan_16_bfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                                  split=DEFAULT_SPLIT,
                                                                  action_order='bfs_randat'),
    'megan_16_random': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=16,
                                                              split=DEFAULT_SPLIT, action_order='random'),

    # variant that we use for USPTO-FULL
    'megan_32_bfs_randat': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=32,
                                                                  split=DEFAULT_SPLIT,
                                                                  action_order='bfs_randat'),

    # variant that we use for USPTO-MIT (forward synthesis prediction)
    'megan_for_8_dfs_cano': lambda: MeganTrainingSamplesFeaturizer(n_jobs=N_JOBS, max_n_steps=8,
                                                                   split=DEFAULT_SPLIT, forward=True,
                                                                   action_order='dfs'),
}


def get_dataset(dataset_key: str) -> Dataset:
    """
    :param: dataset_key: key of a Dataset
    :return: a Dataset for a specified key
    """
    if dataset_key not in DATASET_INITIALIZERS:
        raise ValueError(f"No dataset for key {dataset_key}")

    return DATASET_INITIALIZERS[dataset_key]()


def get_split(split_key: str, **kwargs) -> DatasetSplit:
    """
    :param: split_key: key of a DatasetSplit
    :param: kwargs: additional keyword arguments
    :return: a DatasetSplit for a specified key
    """
    if split_key not in SPLIT_INITIALIZERS:
        raise ValueError(f"No split for key {split_key}")

    return SPLIT_INITIALIZERS[split_key](**kwargs)


def get_featurizer(featurizer_key: str) -> ReactionFeaturizer:
    """
    :param: featurizer_key: key of a ReactionFeaturizer
    :return: a ReactionFeaturizer for a specified key
    """
    if featurizer_key not in FEATURIZER_INITIALIZERS:
        raise ValueError(f"No featurizer for key {featurizer_key}")

    return FEATURIZER_INITIALIZERS[featurizer_key]()
