# -*- coding: utf-8 -*-
"""
A dataset containing 50k reactions of 10 types from USPTO data. It is commonly used in papers.
"""
import logging
import os

import pandas as pd
from tqdm import tqdm

from src import DATA_DIR
from src.datasets import Dataset
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

    def acquire(self):
        x = {
            'product': [],
            'substrates': []
        }
        split = {
            'train': [],
            'valid': [],
            'test': []
        }
        meta = {
            'reaction_type_id': [],
            'id': []
        }

        for split_key, filename in (('train', 'raw_train.csv'), ('valid', 'raw_val.csv'), ('test', 'raw_test.csv')):
            data_path = os.path.join(DATA_DIR, f'uspto_50k/{filename}')
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f'File not found at: {data_path}. Please download data manually from '
                    'https://www.dropbox.com/sh/6ideflxcakrak10/AAAESdZq7Y0aNGWQmqCEMlcza/typed_schneider50k '
                    'and extract to the required location.')
            data_df = pd.read_csv(data_path)

            for reaction_smiles in tqdm(data_df['reactants>reagents>production'], total=len(data_df),
                                        desc="generating product/substrates SMILES'"):
                subs, prod = tuple(reaction_smiles.split('>>'))
                subs, prod = complete_mappings(subs, prod)
                x['substrates'].append(subs)
                x['product'].append(prod)

            for split_key2 in ['train', 'valid', 'test']:
                if split_key == split_key2:
                    split[split_key2] += [1 for _ in range(len(data_df))]
                else:
                    split[split_key2] += [0 for _ in range(len(data_df))]

            meta['reaction_type_id'] += data_df['class'].tolist()
            meta['id'] += data_df['id'].tolist()

        logger.info(f"Saving 'x' to {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

        logger.info(f"Saving {self.metadata_path}")
        pd.DataFrame(meta).to_csv(self.metadata_path, sep='\t')

        split_path = os.path.join(self.dir, 'default_split.csv')
        logger.info(f"Saving default split to {split_path}")
        pd.DataFrame(split).to_csv(split_path)

