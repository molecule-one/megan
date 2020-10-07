# -*- coding: utf-8 -*-
"""
Utility functions for datasets
"""
import gzip
import logging
import os
import shutil
import zipfile

import requests

logger = logging.getLogger(__name__)


def unzip_and_clean(archive_dir: str, file_name: str):
    archive_path = os.path.join(archive_dir, file_name)

    if file_name.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as f:
            f.extractall(path=archive_dir)
    elif file_name.endswith('.gz'):
        output_path = archive_path.replace('.gz', '')
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f'Unsupported archive format for file: {archive_path}')

    os.remove(archive_path)


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
