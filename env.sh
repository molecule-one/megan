#!/usr/bin/env bash

CONDA_ENV_NAME=megan
export PROJECT_ROOT=`pwd`
export DATA_DIR=$PROJECT_ROOT/data

export CONFIGS_DIR=./configs
export LOGS_DIR=./logs
export MODELS_DIR=./models

# random seed we used in training
export RANDOM_SEED=132435

echo "Project root set as $PROJECT_ROOT"

conda_path(){
	export PYTHONPATH=
	export PYTHONNOUSERSITE=1
}

# This is a reasonable default. It isolates conda env from system packages
# Remove if causes any problems (this might mean conda updates its linking)
conda_path $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# set this variable to use other GPU than 0
export CUDA_VISIBLE_DEVICES=0

# number of jobs in multithreaded parts of code. -1 <=> all available CPU cores
export N_JOBS=-1

