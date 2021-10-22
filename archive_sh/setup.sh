#!/bin/bash

echo ">>>>> Cloning repository from Github"
git clone https://github.com/madenindya/newlm.git
cd newlm
git fetch origin -a
git checkout v0.1.1
mkdir examples/data
mkdir examples/data/text

echo ">>>>> Create conda environment `newlm-py38`"
conda create -n newlm-py38 python=3.8.8
source ~/anaconda3/etc/profile.d/conda.sh # change if necessary
conda activate newlm-py38
conda info

echo ">>>>> Install packages for newlm"
pip install -r requirements.txt

echo ">>>>> Login WandB"
wandb login

# # download data
# echo ">>>>> Download the data"
# # ln -s /mnt///newlm/examples/data/wikipedia examples/data
# # ln -s /mnt///newlm/examples/data/books examples/data

# # Create data for experiment
# ./setup_data.sh

echo ">>>>> Finish Setup (repo, conda, pip, wandb)"
