# -*- coding: utf-8 -*-
"""
A dataset containing 50k reactions of 10 types from USPTO data. It is commonly used in papers.
"""
import logging
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.datasets import Dataset
from src.datasets.util import download_url
from src.utils import complete_mappings

logger = logging.getLogger(__name__)


REACTION_TYPES = {
    1: 'heteroatom alkylation and arylation',
    2: 'acylation and related processes',
    3: 'C-C bond formation',
    4: 'heterocycle formation',
    5: 'protections',
    6: 'deprotections',
    7: 'reductions',
    8: 'oxidations',
    9: 'functional group interconversion (FGI)',
    10: 'functional group addition (FGA)'
}


class Uspto50k(Dataset):
    def __init__(self):
        super(Uspto50k, self).__init__()
        self.raw_data_path = os.path.join(self.feat_dir, 'data_processed.csv')

    @property
    def meta_info(self) -> dict:
        return {'reaction_types': REACTION_TYPES, 'max_n_nodes': 100}

    @property
    def key(self) -> str:
        return 'uspto_50k'

    def _download(self):
        url = "https://raw.githubusercontent.com/connorcoley/retrosim/master/retrosim/data/data_processed.csv"
        logger.info(f"Downloading raw data from {url}")
        download_url(url, self.raw_data_path)
        logger.info(f"File downloaded and unpacked to {self.feat_dir}")

    def _preprocess(self):
        data_df = pd.read_csv(self.raw_data_path)

        x = {
            'product': [],
            'substrates': []
        }

        for reaction_smiles in tqdm(data_df['rxn_smiles'], total=len(data_df),
                                    desc="generating product/substrates SMILES'"):
            subs, prod = tuple(reaction_smiles.split('>>'))
            subs, prod = complete_mappings(subs, prod)
            x['substrates'].append(subs)
            x['product'].append(prod)

        # generate 'default' split
        split_fracs = [0.8, 0.1, 0.1]
        split_inds = list(range(len(split_fracs)))

        split_ind = np.random.choice(split_inds, p=split_fracs, size=len(data_df))
        split = {
            'train': (split_ind == 0).astype(int),
            'valid': (split_ind == 1).astype(int),
            'test': (split_ind == 2).astype(int),
        }

        meta = {
            'reaction_type_id': data_df['class'],
            'patent_id': data_df['id']
        }

        logger.info(f"Saving 'x' to {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

        logger.info(f"Saving {self.metadata_path}")
        pd.DataFrame(meta).to_csv(self.metadata_path, sep='\t')

        split_path = os.path.join(self.dir, 'default_split.csv')
        logger.info(f"Saving default split to {split_path}")
        pd.DataFrame(split).to_csv(split_path)

    def acquire(self):
        self._download()
        self._preprocess()
