#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Featurize a dataset using a selected method

python bin/featurize.py dataset_key featurizer
"""
import logging

import argh

from src.config import DATASET_INITIALIZERS, FEATURIZER_INITIALIZERS, get_featurizer, get_dataset

logger = logging.getLogger(__name__)


@argh.arg('dataset_key', choices=list(DATASET_INITIALIZERS.keys()))
@argh.arg('featurizer_key', choices=list(FEATURIZER_INITIALIZERS.keys()))
def featurize(dataset_key: str, featurizer_key: str):
    """
    Featurize dataset using a selected method

    :param dataset_key: key of the dataset
    :param featurizer_key: key of the dataset
    """
    dataset = get_dataset(dataset_key)
    featurizer = get_featurizer(featurizer_key)
    logger.info(f"Featurizing with '{featurizer_key}' on dataset '{dataset_key}'")
    featurizer.featurize_dataset(dataset)
    logger.info(f"Finished featurizing with '{featurizer_key}' on dataset '{dataset_key}'!")


if __name__ == '__main__':
    argh.dispatch_command(featurize)
