"""
Function for finding all possible values of atom/bond features in a dataset of reactions
"""

import logging
import os

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from src.feat.graph_features import ATOM_PROPS, try_get_atom_feature, BOND_PROPS, try_get_bond_feature
from src.feat.utils import fix_explicit_hs

logger = logging.getLogger(__name__)

DEBUG = int(os.environ.get('DEBUG', 0))


def update_feat_values(mol: Mol, atom_props: dict, bond_props: dict):
    for atom in mol.GetAtoms():
        for prop_key in atom_props.keys():
            atom_props[prop_key].add(try_get_atom_feature(atom, prop_key))

    for bond in mol.GetBonds():
        for prop_key in bond_props.keys():
            bond_props[prop_key].add(try_get_bond_feature(bond, prop_key))


def find_properties_parallel(params) -> dict:
    thread_num, data_x, feat_loop = params

    atom_props, bond_props = {}, {}
    for key in ATOM_PROPS.keys():
        atom_props[key] = set()
    for key in BOND_PROPS.keys():
        bond_props[key] = set()

    for sub_smi, prod_smi in feat_loop(zip(data_x['substrates'], data_x['product']),
                                       desc='Thread {}: finding all possible feature values...'.format(thread_num),
                                       total=len(data_x['substrates'])):
        for smi in (sub_smi, prod_smi):
            # noinspection PyBroadException
            try:
                mol = Chem.MolFromSmiles(smi)
            except:
                continue
            if mol is None:
                continue

            try:
                mol = fix_explicit_hs(mol)
                update_feat_values(mol, atom_props, bond_props)
            except Exception as e:
                if DEBUG >= 0:
                    logger.warning(f"Exception while 'update_feat_values': {str(e)}")
                continue

    return {'atom': atom_props, 'bond': bond_props}

