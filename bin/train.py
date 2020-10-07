#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MEGAN. Call as:

python bin/train.py config_path model_dir

"""
import logging
import os
from shutil import copyfile

import tensorflow as tf

# shut down logs from tensorboard
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# this makes tensorflow use CPU, which saves some GPU RAM
tf.config.experimental.set_visible_devices([], 'GPU')

from collections import Counter

import gin
import numpy as np
import torch
from src.config import get_featurizer
from tqdm import tqdm

from src.get_instances import get_dataset
from src.model.utils import DumpTensorflowSummaries
from src.model.megan import Megan
from src.model.megan_utils import generate_batch
from src.utils import dispatch_utils, summary, save_weights, load_state_dict
from src.utils.dispatch_utils import save_current_config, log_current_config
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer, get_actions_vocab_path, get_prop2oh_vocab_path

logger = logging.getLogger(__name__)


@gin.configurable(blacklist=['save_path'])
def train_megan(
        save_path: str,
        featurizer_key: str,
        learning_rate: float = 0.0001,
        train_samples_per_epoch: int = -1,
        valid_samples_per_epoch: int = -1,
        batch_size: int = 4,
        gen_lr_factor: float = 0.1,
        gen_lr_patience: int = 4,
        big_lr_epochs: int = -1,
        early_stopping: int = 16,
        start_epoch: int = 0,
        megan_warmup_epochs: int = 1,
        save_each_epoch: bool = False,
        max_n_epochs: int = 1000):
    """
    Train MEGAN model
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoints_path = os.path.join(save_path, 'checkpoints')
    if save_each_epoch and not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    log_current_config()
    conf_path = os.path.join(save_path, 'config.gin')
    save_current_config(conf_path)

    model_path = os.path.join(save_path, 'model.pt')
    best_model_path = os.path.join(save_path, 'model_best.pt')

    summary_dir = 'summary'
    summary_dir = os.path.join(save_path, summary_dir)
    tf_callback = DumpTensorflowSummaries(save_path=summary_dir, step_multiplier=train_samples_per_epoch)

    dataset = get_dataset()
    featurizer = get_featurizer(featurizer_key)
    assert isinstance(featurizer, MeganTrainingSamplesFeaturizer)
    action_vocab = featurizer.get_actions_vocabulary(dataset.feat_dir)

    # copy featurizer dictionary files needed for using the model
    feat_dir = featurizer.dir(dataset.feat_dir)
    model_feat_dir = featurizer.dir(save_path)
    if not os.path.exists(model_feat_dir):
        os.makedirs(model_feat_dir)
    copyfile(get_actions_vocab_path(feat_dir), get_actions_vocab_path(model_feat_dir))
    copyfile(get_prop2oh_vocab_path(feat_dir), get_prop2oh_vocab_path(model_feat_dir))

    logger.info("Creating model...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Megan(n_atom_actions=action_vocab['n_atom_actions'],
                  n_bond_actions=action_vocab['n_bond_actions'],
                  prop2oh=action_vocab['prop2oh']).to(device)
    summary(model)

    logger.info("Loading data...")
    data_dict = {}

    logger.info(f"Training for maximum of {max_n_epochs} epochs...")

    start_learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def set_lr(lr: float):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run_batch(ind: np.ndarray, train: bool) -> dict:
        if train:
            optimizer.zero_grad()

        batch_ind = np.random.choice(ind, size=batch_size, replace=False)
        batch_metrics = {}
        batch = generate_batch(batch_ind, data_dict['metadata'], featurizer, data_dict['data'], action_vocab=action_vocab)

        batch_result = model(batch)
        actions = batch_result['output']

        target, n_steps = batch['target'], batch['n_steps']
        n_total_steps = torch.sum(n_steps)

        y_max_pred_prob, y_pred = torch.max(actions, dim=-1)
        y_val, y_true = torch.max(target, dim=-1)
        y_val_one = y_val == 1
        is_hard = batch['is_hard']

        weight = torch.ones_like(is_hard)
        avg_weight = torch.mean(weight.float(), axis=-1)

        weight = weight * y_val_one
        weight = weight.unsqueeze(-1).expand(*actions.shape)
        target_one = target == 1
        eps = 1e-09

        loss = -torch.log2(actions + ~target_one + eps) * target_one * weight
        loss = torch.sum(loss, dim=-1)
        path_losses = torch.sum(loss, dim=-1) / (avg_weight * 16)

        min_losses = []
        # for each reaction, use the minimum loss for each possible path as the loss to optimize
        path_i = 0
        for n_paths in batch['n_paths']:
            path_loss = torch.min(path_losses[path_i: path_i + n_paths])
            min_losses.append(path_loss.unsqueeze(-1))
            path_i += n_paths
        min_losses = torch.cat(min_losses)

        loss = torch.mean(min_losses)

        if torch.isinf(loss):
            raise ValueError('Infinite loss (correct action has predicted probability=0.0)')

        if loss != loss:  # this is only true for NaN in pytorch
            raise ValueError('NaN loss')

        # skip accuracy metrics if there are no positive samples in batch
        correct = ((y_pred == y_true) & y_val_one).float()

        step_correct = torch.sum(correct) / n_total_steps
        batch_metrics['step_acc'] = step_correct.cpu().detach().numpy()

        total_hard = torch.sum(is_hard)
        if total_hard > 0:
            hard_correct = torch.sum(correct * is_hard) / total_hard
            batch_metrics['step_acc_hard'] = hard_correct.cpu().detach().numpy()

        is_easy = (1.0 - is_hard) * y_val_one

        total_easy = torch.sum(is_easy)
        if total_easy > 0:
            easy_correct = torch.sum(correct * is_easy) / total_easy
            batch_metrics['step_acc_easy'] = easy_correct.cpu().detach().numpy()

        all_correct = torch.sum(correct, dim=-1)
        all_correct = all_correct == n_steps
        acc = []
        path_i = 0
        for n_paths in batch['n_paths']:
            corr = any(all_correct[i] == 1 for i in range(path_i, path_i + n_paths))
            acc.append(corr)
            path_i += n_paths
        if len(acc) > 0:
            batch_metrics['acc'] = np.mean(acc)

        if train:
            loss.backward()
            optimizer.step()

        batch_metrics['loss'] = loss.cpu().detach().numpy()
        return batch_metrics

    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run_epoch(set_key: str, i_ep: int, all_ind: np.ndarray, train: bool, batches_per_epoch: int,
                  lr_step: float = 0.0):
        torch.cuda.empty_cache()
        if train:
            model.train()
        else:
            model.eval()

        metrics = {}
        counts = Counter()

        for batch_i in tqdm(range(batches_per_epoch), desc=f'{save_path} {set_key} epoch {i_ep + 1}'):
            if lr_step > 0:
                set_lr(get_lr() + lr_step)
            try:
                batch_metrics = run_batch(all_ind, train)
                for k, v in batch_metrics.items():
                    if k not in metrics:
                        metrics[k] = 0
                    metrics[k] += v
                    counts[k] += 1
            except AssertionError as e:
                # batch skipped because of zero loss
                logger.debug(f"Exception while running batch: {str(e)}")
            except Exception as e:
                logger.warning(f"Exception while running batch: {str(e)}")
                raise e

        metrics = dict((k, v / counts[k]) for k, v in metrics.items())
        str_metrics = ', '.join("{:s}={:.4f}".format(k, v) for k, v in metrics.items())
        logger.info(f'{set_key} epoch {i_ep + 1}: {str_metrics}')

        if train:
            save_weights(model_path, model, optimizer, epoch=i_ep, lr=get_lr(), no_progress=no_progress)

            if save_each_epoch:
                model_epoch_path = os.path.join(checkpoints_path, f'model_{(i_ep + 1) * train_samples_per_epoch}.pt')
                save_weights(model_epoch_path, model, optimizer, epoch=i_ep, lr=get_lr())
        return metrics

    best_acc = 0
    no_progress = 0

    if os.path.exists(model_path):
        checkpoint = load_state_dict(model_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        logger.info("Resuming training after {} epochs".format(start_epoch))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr' in checkpoint:
            learning_rate = checkpoint['lr']
            start_learning_rate = learning_rate
            logger.info("Resuming training with LR={:f} epochs".format(learning_rate))
            set_lr(learning_rate)
        if 'valid_acc' in checkpoint:
            best_acc = checkpoint['valid_acc']
            logger.info(f"Best acc so far: {best_acc}")

    megan_warmup_epochs = max(megan_warmup_epochs - start_epoch, 0)
    if megan_warmup_epochs > 0:
        learning_rate = 0.0
        set_lr(learning_rate)

    no_progress = 0
    no_progress_lr = 0

    logger.info('Loading data')
    loaded_data = featurizer.load(dataset.feat_dir)
    chunk_metadata = loaded_data['reaction_metadata']
    data_dict['data'] = loaded_data
    data_dict['metadata'] = chunk_metadata
    data_dict['mean_n_steps'] = np.mean(data_dict['metadata']['n_samples'])

    metadata = data_dict['metadata']
    if 'remapped' in metadata:
        train_ind = (metadata['is_train'] == 1) & (metadata['remapped'])
        valid_ind = (metadata['is_train'] == 0) & (metadata['remapped'])
    else:
        train_ind = metadata['is_train'] == 1
        valid_ind = metadata['is_train'] == 0

    if 'path_i' in metadata:
        train_ind = train_ind & (metadata['path_i'] == 0)
        valid_ind = valid_ind & (metadata['path_i'] == 0)

    train_ind = np.argwhere(train_ind).flatten()
    valid_ind = np.argwhere(valid_ind).flatten()

    logger.info(f"Training on chunk of {len(train_ind)} training samples and {len(valid_ind)} valid samples")
    if train_samples_per_epoch == -1:
        train_samples_per_epoch = len(train_ind)
    if valid_samples_per_epoch == -1:
        valid_samples_per_epoch = len(valid_ind)
    train_batches_per_epoch = int(np.ceil(train_samples_per_epoch / batch_size))
    valid_batches_per_epoch = int(np.ceil(valid_samples_per_epoch / batch_size))

    logger.info(f'Starting training on epoch {start_epoch + 1} with Learning Rate={learning_rate} '
                f'({megan_warmup_epochs} warmup epochs)')

    for epoch_i in range(start_epoch, max_n_epochs):
        if epoch_i == megan_warmup_epochs:
            set_lr(start_learning_rate)
            logger.info(f'Learning rate set to {start_learning_rate} after {megan_warmup_epochs} warmup epochs')

        if big_lr_epochs != -1 and epoch_i == big_lr_epochs:
            learning_rate *= gen_lr_factor
            no_progress = 0
            no_progress_lr = 0
            set_lr(learning_rate)
            logger.info(f'Changing Learning Rate to {learning_rate}')

        if megan_warmup_epochs > 0:
            warmup_lr_step = start_learning_rate / (train_batches_per_epoch * megan_warmup_epochs)
        else:
            warmup_lr_step = 0

        learning_rate = get_lr()
        train_metrics = run_epoch('train', epoch_i, train_ind, True, train_batches_per_epoch,
                                  lr_step=warmup_lr_step if epoch_i < megan_warmup_epochs else 0.0)
        with torch.no_grad():
            valid_metrics = run_epoch('valid', epoch_i, valid_ind, False, valid_batches_per_epoch)

        all_metrics = {}
        for key, val in train_metrics.items():
            all_metrics[f'train_{key}'] = val
        for key, val in valid_metrics.items():
            all_metrics[f'valid_{key}'] = val

        all_metrics['lr'] = learning_rate
        tf_callback.on_epoch_end(epoch_i + 1, all_metrics)

        valid_acc = valid_metrics['acc']
        if valid_acc > best_acc:
            logger.info(f'Saving best model from epoch {epoch_i + 1} to {best_model_path} (acc={valid_acc})')
            save_weights(best_model_path, model, optimizer, epoch=epoch_i, lr=learning_rate, valid_acc=valid_acc)

            best_acc = valid_acc
            no_progress = 0
            no_progress_lr = 0
        else:
            no_progress += 1
            no_progress_lr += 1

        if big_lr_epochs == -1 or epoch_i >= big_lr_epochs:
            if no_progress_lr > gen_lr_patience:
                learning_rate *= gen_lr_factor
                logger.info(f'Changing Learning Rate to {learning_rate}')
                set_lr(learning_rate)
                no_progress_lr = 0

            if no_progress > early_stopping:
                logger.info(f'Early stopping after {epoch_i + 1} epochs')
                break

    logger.info("Experiment finished!")


if __name__ == "__main__":
    dispatch_utils.dispatch_configurable_command(train_megan)
