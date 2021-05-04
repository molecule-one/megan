#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MEGAN model.

python bin/eval.py model_dir
"""

import os
import time
from collections import Counter
from typing import Tuple

from rdkit.Chem import Mol

from src import config
from src.config import get_featurizer
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging

import argh
import gin
import numpy as np
import torch
from src.utils import mol_to_unmapped, mol_to_unmapped_smiles, mark_reactants
from rdkit import Chem
from tqdm import tqdm
from src.feat.utils import fix_explicit_hs, add_map_numbers
from itertools import islice

# noinspection PyUnresolvedReferences
from bin.train import train_megan
from src.model.megan import Megan
from src.model.beam_search import beam_search
from src.model.megan_utils import get_base_action_masks, RdkitCache
from src.utils import load_state_dict
from src.utils.dispatch_utils import run_with_redirection

logger = logging.getLogger(__name__)


def prediction_is_correct(y_pred: str, y_true: str):
    if not y_pred:
        return False

    # order of compounds in SMILES does not matter
    pred_mols = Counter(y_pred.split('.'))
    true_mols = Counter(y_true.split('.'))

    for mol_smi in true_mols:
        if pred_mols[mol_smi] != true_mols[mol_smi]:
            return False
    return True


def remap_reaction_to_canonical(input_mol: Mol, target_mol: Mol) -> Tuple[Mol, Mol]:
    """
    Re-maps reaction according to order of atoms in RdKit - this makes sure that stereochemical SMILES are canonical.
    Note: this method does not transfer any information from target molecule to the input molecule
    (the input molecule is mapped according to its order of atoms in its canonical SMILES)
    """

    # converting Mol to smiles and again to Mol makes atom order canonical
    input_mol = Chem.MolFromSmiles(Chem.MolToSmiles(input_mol))
    target_mol = Chem.MolFromSmiles(Chem.MolToSmiles(target_mol))

    map2map = {}
    for i, a in enumerate(input_mol.GetAtoms()):
        map2map[int(a.GetAtomMapNum())] = i + 1
        a.SetAtomMapNum(i + 1)

    max_map = max(map2map.values())

    for i, a in enumerate(target_mol.GetAtoms()):
        old_map = int(a.GetAtomMapNum())
        if old_map in map2map:
            new_map = map2map[old_map]
        else:
            new_map = max_map + 1
            max_map += 1
        a.SetAtomMapNum(new_map)

    return input_mol, target_mol


def evaluate_megan(save_path: str, beam_size: int = 10, max_gen_steps: int = 16, beam_batch_size: int = 10,
                   show_every: int = 100, n_max_atoms: int = 200,
                   dataset_key: str = 'uspto_50k', split_type: str = 'default',
                   split_key: str = 'test', results_file: str = ''):
    """
    Evaluate MEGAN model
    """
    config_path = os.path.join(save_path, 'config.gin')
    gin.parse_config_file(config_path)

    dataset = config.get_dataset(dataset_key)
    featurizer_key = gin.query_parameter('train_megan.featurizer_key')
    featurizer = get_featurizer(featurizer_key)
    assert isinstance(featurizer, MeganTrainingSamplesFeaturizer)
    action_vocab = featurizer.get_actions_vocabulary(save_path)

    base_action_masks = get_base_action_masks(n_max_atoms + 1, action_vocab=action_vocab)

    logger.info("Creating model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    predict_forward = featurizer.forward

    logger.info("Loading data...")
    x_df = dataset.load_x()
    meta_df = dataset.load_metadata()
    split_df = config.get_split(split_type).load(dataset.dir)

    model_path = os.path.join(save_path, 'model_best.pt')
    checkpoint = load_state_dict(model_path)
    model = Megan(n_atom_actions=action_vocab['n_atom_actions'], n_bond_actions=action_vocab['n_bond_actions'],
                  prop2oh=action_vocab['prop2oh']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    split_ind = np.argwhere(split_df[split_key] == 1).flatten()
    if 'class' in meta_df:
        split_ind = np.argwhere((split_df[split_key] == 1) & (meta_df['class'] == 1)).flatten()

    np.random.shuffle(split_ind)
    logger.info(f"Evaluating on {len(split_ind)} samples from {split_key}")

    top_k = np.zeros(beam_size, dtype=float)
    accs = np.zeros(beam_size, dtype=float)

    def split_every(n, iterable):
        i = iter(iterable)
        piece = list(islice(i, n))
        while piece:
            yield piece
            piece = list(islice(i, n))

    n_batches = int(np.ceil(len(split_ind)) / beam_batch_size)
    prog_bar = tqdm(desc=f'{save_path} beam search on {split_key}', total=len(split_ind))

    start_time = time.time()
    n_samples, n_gen_reactions = 0, 0
    is_incorrect, is_duplicate = [], []
    n_preds = np.zeros(len(split_ind), dtype=int)

    pred_path = os.path.join(save_path, f'pred_{split_key}_{beam_size}_{max_gen_steps}.txt')
    pred_path_i = 1
    while os.path.exists(pred_path):
        pred_path = os.path.join(save_path, f'pred_{split_key}_{beam_size}_{max_gen_steps}_{pred_path_i}.txt')
        pred_path_i += 1
    target_sub_key = 'reactants' if 'reactants' in x_df else 'substrates'

    for batch_i, batch_ind in enumerate(split_every(beam_batch_size, split_ind)):
        input_mols = []
        target_mols = []

        for ind in batch_ind:
            if predict_forward:
                input_mapped = x_df['substrates'][ind]
                target_mapped = x_df['product'][ind]
            else:
                input_mapped = x_df['product'][ind]
                target_mapped = x_df[target_sub_key][ind]

            try:
                target_mol = Chem.MolFromSmiles(target_mapped)
                input_mol = Chem.MolFromSmiles(input_mapped)

                # mark reactants (this is used only by models that use such information)
                if featurizer.forward:
                    mark_reactants(input_mol, target_mol)

                # remap input and target molecules according to canonical SMILES atom order
                input_mol, target_mol = remap_reaction_to_canonical(input_mol, target_mol)

                # fix a bug in marking explicit Hydrogen atoms by RdKit
                input_mol = fix_explicit_hs(input_mol)

            except Exception as e:
                logger.warning(f'Exception while input mol to SMILES {str(e)}')
                input_mol = None
                target_mol = None

            if input_mol is None or target_mol is None:
                input_mols.append(None)
                target_mols.append(None)
                continue

            input_mols.append(input_mol)
            target_mols.append(target_mol)

        if 'reaction_type_id' in meta_df:
            reaction_types = meta_df['reaction_type_id'][batch_ind].values
        else:
            reaction_types = None

        rdkit_cache = RdkitCache(props=action_vocab['props'])

        with torch.no_grad():
            beam_search_results = beam_search([model], input_mols, rdkit_cache=rdkit_cache, max_steps=max_gen_steps,
                                              beam_size=beam_size, batch_size=beam_batch_size,
                                              base_action_masks=base_action_masks, max_atoms=n_max_atoms,
                                              reaction_types=reaction_types,
                                              action_vocab=action_vocab)

        with open(pred_path, 'a') as fp:
            for sample_i, ind in enumerate(batch_ind):
                input_mol, target_mol = input_mols[sample_i], target_mols[sample_i]
                try:
                    target_smi = mol_to_unmapped_smiles(target_mol)
                    target_mapped = Chem.MolToSmiles(target_mol)
                except Exception as e:
                    logger.info(f"Exception while target to smi: {str(e)}")
                    n_samples += 1
                    continue

                has_correct = False
                final_smis = set()

                results = beam_search_results[sample_i]
                n_preds[n_samples] = len(results)

                fp.write(f'{ind} {Chem.MolToSmiles(input_mol)} {target_smi} {target_mapped}\n')

                for i, path in enumerate(results):
                    if path['final_smi_unmapped']:
                        try:
                            final_mol = Chem.MolFromSmiles(path['final_smi_unmapped'])

                            if final_mol is None:
                                final_smi = path['final_smi_unmapped']
                            else:
                                input_mol, final_mol = remap_reaction_to_canonical(input_mol, final_mol)
                                final_smi = mol_to_unmapped_smiles(final_mol)

                        except Exception as e:
                            final_smi = path['final_smi_unmapped']
                    else:
                        final_smi = path['final_smi_unmapped']

                    # for forward prediction, if we generate more than 1 product we heuristically select the biggest one
                    if predict_forward:
                        final_smi = list(sorted(final_smi.split('.'), key=len))
                        final_smi = final_smi[-1]

                    str_actions = '|'.join(f"({str(a)};{p})" for a, p in path['actions'])
                    str_ch = '{' + ','.join([str(c) for c in path['changed_atoms']]) + '}'
                    fp.write(f'{i} {path["final_smi"]} {final_smi} {str_ch} {str_actions}\n')
                    is_duplicate.append(final_smi in final_smis)
                    is_incorrect.append(final_smi is None or final_smi == '')
                    final_smis.add(final_smi)
                    correct = prediction_is_correct(final_smi, target_smi)
                    # correct = final_smi == target_smi
                    if correct and not has_correct:
                        top_k[i:] += 1
                        accs[i] += 1
                        has_correct = True
                    n_gen_reactions += 1
                fp.write('\n')
                n_samples += 1

        if (batch_i > 0 and batch_i % show_every == 0) or batch_i >= n_batches - 1:
            print("^" * 100)
            print(f'Beam search parameters: beam size={beam_size}, max steps={max_gen_steps}')
            print()
            for k, top in enumerate(top_k):
                acc = accs[k]
                print('Top {:3d}: {:7.4f}% cum {:7.4f}%'.format(k + 1, acc * 100 / n_samples, top * 100 / n_samples))
            print()
            avg_incorrect = '{:.4f}%'.format(100 * np.sum(is_incorrect) / len(is_incorrect))
            avg_duplicates = '{:.4f}%'.format(100 * np.sum(is_duplicate) / len(is_duplicate))
            avg_n_preds = '{:.4f}'.format(n_gen_reactions / n_samples)
            less_preds = '{:.4f}%'.format(100 * np.sum(n_preds[:n_samples] < beam_size) / n_samples)
            zero_preds = '{:.4f}%'.format(100 * np.sum(n_preds[:n_samples] == 0) / n_samples)
            print(f'Avg incorrect reactions in Top {beam_size}: {avg_incorrect}')
            print(f'Avg duplicate reactions in Top {beam_size}: {avg_duplicates}')
            print(f'Avg number of predictions per target: {avg_n_preds}')
            print(f'Targets with < {beam_size} predictions: {less_preds}')
            print(f'Targets with zero predictions: {zero_preds}')
            print()

        prog_bar.update(len(batch_ind))
    prog_bar.close()

    total_time = time.time() - start_time
    s_targets = '{:.4f}'.format(total_time / n_samples)
    s_reactions = '{:.4f}'.format(n_gen_reactions / total_time)
    total_time = '{:.4f}'.format(total_time)
    avg_incorrect = '{:.4f}%'.format(100 * np.sum(is_incorrect) / len(is_incorrect))
    avg_duplicates = '{:.4f}%'.format(100 * np.sum(is_duplicate) / len(is_duplicate))
    avg_n_preds = '{:.4f}'.format(n_gen_reactions / n_samples)
    less_preds = '{:.4f}%'.format(100 * np.sum(n_preds < beam_size) / n_samples)
    zero_preds = '{:.4f}%'.format(100 * np.sum(n_preds == 0) / n_samples)

    summary_path = \
        os.path.join(save_path, f'eval_{split_key}_{beam_size}_{max_gen_steps}.txt')

    with open(summary_path, 'w') as fp:
        fp.write(f'Evaluation on {split_key} set ({split_type} split)\n')
        fp.write(f'Beam size = {beam_size}, batch size = {beam_batch_size}, '
                 f'max gen steps = {max_gen_steps}\n')
        fp.write(f'Avg incorrect reactions in Top {beam_size}: {avg_incorrect}\n')
        fp.write(f'Avg duplicate reactions in Top {beam_size}: {avg_duplicates}\n')
        fp.write(f'Avg number of predictions per target: {avg_n_preds}\n')
        fp.write(f'Targets with < {beam_size} predictions: {less_preds}\n')
        fp.write(f'Targets with zero predictions: {zero_preds}\n')
        fp.write(f'Total evaluation time on {len(split_ind)} targets: {total_time} seconds '
                 f'({s_targets} seconds per target, {s_reactions} reactions per second) \n\n')
        for k, top in enumerate(top_k):
            acc = accs[k]
            fp.write('Top {:3d}: {:7.4f}% cum {:7.4f}%\n'.format(k + 1, acc * 100 / n_samples, top * 100 / n_samples))

    if results_file:
        top_k_str = ' '.join('{:7.2f}%'.format(top * 100 / n_samples) for top in top_k)
        with open(results_file, 'a') as fp:
            fp.write('{:>50s}: {:<s}\n'.format(save_path, top_k_str))

    logger.info(f'Saved Top {beam_size} to {summary_path}')


if __name__ == '__main__':
    run_with_redirection(argh.dispatch_command)(evaluate_megan)
