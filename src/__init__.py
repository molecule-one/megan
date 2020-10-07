# project initialization

import logging
import multiprocessing
import os
import random
import numpy as np

from src.utils import configure_logger

DATA_DIR = os.environ.get("DATA_DIR", './data')
LOGS_DIR = os.environ.get("LOGS_DIR", './logs')
CONFIGS_DIR = os.environ.get("CONFIG_DIR", os.path.join(os.environ['PROJECT_ROOT'], "configs"))
N_JOBS = int(os.environ.get("N_JOBS", -1))

if N_JOBS == -1:
    N_JOBS = multiprocessing.cpu_count()

if int(os.environ.get("DEBUG", 0)) >= 1:
    LOG_LEVEL = logging.DEBUG
else:
    LOG_LEVEL = logging.INFO

configure_logger(name='', console_logging_level=LOG_LEVEL, logs_dir=LOGS_DIR)
logger = logging.getLogger(__name__)


def set_random_seed():
    seed = int(os.environ.get('RANDOM_SEED', 0))
    logger.info(f"Setting random seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


set_random_seed()
