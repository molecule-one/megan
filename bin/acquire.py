#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download, generate and preprocess datasets

python bin/acquire.py dataset_key
"""
import logging

import argh

from src.config import DATASET_INITIALIZERS, get_dataset

logger = logging.getLogger(__name__)


@argh.arg('dataset_key', choices=list(DATASET_INITIALIZERS.keys()))
def acquire(dataset_key: str):
    """
    Downloads and/or generates and preprocesses a selected dataset.

    :param dataset_key: key of the dataset
    """
    dataset = get_dataset(dataset_key)
    logger.info(f"Acquiring dataset for key {dataset_key}")
    dataset.acquire()
    logger.info(f"Dataset for key {dataset_key} acquired successfully!")


if __name__ == '__main__':
    argh.dispatch_command(acquire)
