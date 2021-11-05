#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate newlm-py38
conda info
wandb login

echo ">>>>> ELMO BERT-Causal 100 Percent"
WANDB_ENTITY=collab-research-kata WANDB_PROJECT=newlm python -m newlm run_pretrain --config_file="examples/configs_gcloud/run-100-percent.elmo-bert-causal.yaml"
