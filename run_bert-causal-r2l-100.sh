#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate newlm-py38
conda info
wandb login

echo ">>>>> BERT-Causal R2L 100 Percent"
WANDB_PROJECT=newlm python -m newlm run_pretrain --config_file="examples/configs_gcloud/run-100-percent.bert-causal-r2l.yaml"
