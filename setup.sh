#!/bin/bash

# download code repository
echo ">>>>> Cloning repository from Github"
git clone https://github.com/madenindya/newlm.git
cd newlm

# download data
echo ">>>>> Download the data"
# # ln -s /mnt///newlm/examples/data/wikipedia examples/data
# # ln -s /mnt///newlm/examples/data/books examples/data

# create conda env
echo ">>>>> Create conda environment `newlm-py38`"
conda create -n newlm-py38 python=3.8.8
source ~/miniconda3/etc/profile.d/conda.sh # change to anaconda3 if necessary
conda activate newlm-py38
conda info

# pip install
echo ">>>>> Install package for newlm"
pip install -r requirements.txt

# wandb login
echo ">>>>> Login WandB"
wandb login

# # Create data for experiment
# ./setup_data.sh

# # Run pretrain with Trial data (optional)
# echo ">>>>> Run Trial"
# source .env.gcloud
# python -m newlm run_all --config_file="examples/configs/run-trial.yaml"
