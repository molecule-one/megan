# -*- coding: utf-8 -*-
"""
Common functions for models
"""
import tensorflow
import torch
from torch.autograd import Variable

from src import logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    if type(y_tensor) != torch.LongTensor:
        y_tensor = y_tensor.type(torch.LongTensor).to(device)
    y_tensor = y_tensor.view(-1, 1)

    n_dims = n_dims + 1 if n_dims is not None else int(torch.max(y_tensor)) + 2
    if y.dim() < 2 and y.sum() == 0:
        y_one_hot = torch.zeros((1, n_dims - 1), device=device)
    else:
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=device).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot[:, 1:]
        y_one_hot = y_one_hot.view(*y.shape, -1)

    return y_one_hot


class DumpTensorflowSummaries:
    def __init__(self, save_path, step_multiplier: int = 1):
        self._save_path = save_path
        self.multiplier = step_multiplier
        super(DumpTensorflowSummaries, self).__init__()

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.summary.create_file_writer(self._save_path)
        return self._file_writer

    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer.as_default():
            for key, value in logs.items():
                try:
                    tensorflow.summary.scalar(key, value, step=epoch * self.multiplier)
                except Exception as e:
                    logger.warning(str(e))
            self.file_writer.flush()
