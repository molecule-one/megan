# -*- coding: utf-8 -*-
"""
USPTO-FULL - large dataset of mapped reactions from https://github.com/Hanjun-Dai/GLN
"""
import logging
import os

import pandas as pd
from src.datasets import Dataset
from src.feat.utils import fix_incomplete_mappings
from src.split.basic_splits import DefaultSplit
from src.utils import filter_reactants
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UsptoFull(Dataset):
    """
    A standard benchmark dataset of ~1 million reactions with stereochemical information.
    """

    def __init__(self):
        super(UsptoFull, self).__init__()
        self.raw_data_dir = os.path.join(self.feat_dir, 'data')
        if not os.path.exists(self.raw_data_dir):
            os.mkdir(self.raw_data_dir)

    @property
    def key(self) -> str:
        return 'uspto_full'

    def acquire(self):
        split_keys = ['train', 'valid', 'test']
        file_names = [f'US_patents_1976-Sep2016_1product_reactions_{k}.csv' for k in split_keys]

        file_paths = [os.path.join(self.raw_data_dir, name) for name in file_names]

        for file_path in file_paths:
            if not os.path.exists(file_path):
                file_paths_str = '\n'.join(file_paths)
                logger.error('Please download data files manually and put them in the following locations:'
                             f'\n{file_paths_str}\n'
                             'Raw data should be found at: https://ibm.ent.box.com/v/ReactionSeq2SeqDataset')
                raise FileNotFoundError(file_path)

        metadata = {
            'id': [],
        }

        x = {
            'substrates': [],
            'reactants': [],
            'reagents': [],
            'product': []
        }

        split = {k: [] for k in split_keys}
        split_path = DefaultSplit().path(self.dir)
        cur_id = 0

        for split_key, file_path in tqdm(zip(split_keys, file_paths),
                                         desc='reading data from splits', total=len(split_keys)):
            n_saved = 0
            split_df = pd.read_csv(file_path, skiprows=2, sep='\t')
            for reaction in tqdm(split_df['OriginalReaction'],
                                 desc=split_key, total=len(split_df)):
                parts = reaction.split('>')
                assert len(parts) == 3  # reactants, reagents, product
                if len(parts[1]) > 0:
                    substrates = '.'.join([parts[0], parts[1]])
                else:
                    substrates = parts[0]
                product = parts[2].split(' ')[0]

                try:
                    sub_mol = Chem.MolFromSmiles(substrates)
                    prod_mol = Chem.MolFromSmiles(product)
                    sub_mol, prod_mol = fix_incomplete_mappings(sub_mol, prod_mol)
                    substrates = Chem.MolToSmiles(sub_mol)
                    product = Chem.MolToSmiles(prod_mol)

                    sub_mols = [Chem.MolFromSmiles(t) for t in Chem.MolToSmiles(sub_mol).split('.')]
                    reactant_mol = filter_reactants(sub_mols, prod_mol)
                    reactants = Chem.MolToSmiles(reactant_mol)
                except:
                    reactants = parts[0]

                if not isinstance(substrates, str) or len(substrates) == 0:
                    substrates = product
                    reactants = product

                x['substrates'].append(substrates)
                x['reactants'].append(reactants)
                x['reagents'].append(parts[1])
                x['product'].append(product)

                metadata['id'].append(cur_id)
                cur_id += 1
                n_saved += 1

                for key in split_keys:
                    if key == split_key:
                        split[key].append(1)
                    else:
                        split[key].append(0)

            logger.info(f'Saved {n_saved} reactions from {split_key}')

        logger.info(f'Collected a total of {len(x["substrates"])} reactions')

        logger.info(f"Saving {self.metadata_path}")
        pd.DataFrame(metadata).to_csv(self.metadata_path, sep='\t')

        logger.info(f"Saving {self.x_path}")
        pd.DataFrame(x).to_csv(self.x_path, sep='\t')

        logger.info(f"Saving {split_path}")
        pd.DataFrame(split).to_csv(split_path)
