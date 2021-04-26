# Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits

Code for *"Molecule Edit Graph Attention Network: Modeling Chemical Reactions as Sequences of Graph Edits"* (https://arxiv.org/abs/2006.15426)

Code was run/tested for:

    - python 3.6
    - pytorch 1.3.1
    - tensorflow 2.0
    - rdkit 2020.03.2

Pytorch is used for building, training and evaluating models. CUDA support is recommended.

Tensorflow is used only for visualizing training process (tensorboard). CUDA support is not required.


### Environment setup
We recommend running MEGAN in an isolated conda environment, which can be created with:

`conda env create -f env.yml`

Edit `env.sh` file so it suits your configuration, if necessary. Before running any scripts, run:

`source env.sh`

This activates the conda environment and sets a few environment values.


### Download training/evaluation data
For USPTO-50k, the data needs to be first manually downloaded from:
https://www.dropbox.com/sh/6ideflxcakrak10/AAAESdZq7Y0aNGWQmqCEMlcza/typed_schneider50k
and unpacked to the `data/uspto_50k` folder
(Thanks to the authors of https://github.com/Hanjun-Dai/GLN for providing the data).

The following scripts download the datasets and generate train/val/test split:
```
python bin/acquire.py uspto_50k  # assumes that raw data is in data/uspto_50k
python bin/acquire.py uspto_mit
python bin/acquire.py uspto_full
```

### Preprocessing training data
The following scripts build graph representation of data needed to train MEGAN:

```
python bin/featurize.py uspto_50k megan_16_bfs_randat
python bin/featurize.py uspto_mit megan_for_8_dfs_cano
python bin/featurize.py uspto_full megan_32_bfs_randat
```

Datasets and featurizers are defined in `src/config.py`.

By default, featurization is multithreaded with number of jobs equal to the number of CPUs. It can be changed by:

`N_JOBS=N python bin/featurize.py uspto_full megan_32_bfs_randat`

where N is an integer >= 1

### Training
```
python bin/train.py uspto_50k models/uspto_50k
python bin/train.py uspto_50k_rt models/uspto_50k_rt
python bin/train.py uspto_mit models/uspto_mit_mix
python bin/train.py uspto_mit_sep models/uspto_mit_sep
```

This trains models with the same configuration as we describe in the paper.

We use gin-config (https://github.com/google/gin-config) for managing training hyperparameters. Gin configuration files are in `configs`. Configuration values can also be passed as script parameters like:

`python bin/train.py uspto_50k models/uspto_50k --learning_rate 0.5 --n_encoder_conv 8`

Training takes from about 10 hours for USPTO-50k to about 60 hours for USPTO-FULL on a single Nvidia GeForce GTX 1070 GPU.

### Evaluation
```
python bin/eval.py models/uspto_50k --beam-size 50 --show-every 100
python bin/eval.py models/uspto_50k_rt --beam-size 50 --show-every 100
python bin/eval.py models/uspto_mit_mix --beam-size 10 --show-every 1000
python bin/eval.py models/uspto_mit_sep --beam-size 10 --show-every 1000
python bin/eval.py models/uspto_full --beam-size 50 --show-every 1000
```

For evaluation script we use `argh`, so `_` in parameter names are replaced with `-`.
Evaluation can take long time, especially for large beam sizes (up to a couple of hours for USPTO-FULL with beam size 50).

Evaluation produces two files: `eval_*.txt` has calculated Top K values, `pred_*.txt` contains predicted SMILES and actions.

### Packed data and models

We include packed pre-processed data, as well as weights of the model trained on USPTO-50k for two variants (reaction type unknown/reaction type given) as a GitHub Release with version number v1.1 in this repo. To use data and pretrained models, unpack the "megan_data.zip" archive in the root directory of the project.
