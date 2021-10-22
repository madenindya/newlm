#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate newlm-py38
conda info

echo ">>>>> BERT-Causal 100 Percent and Run Downstream GLUE"
WANDB_ENTITY=collab-research-kata WANDB_PROJECT=newlm python -m newlm run_all --config_file="examples/configs_gcloud/run-bc.100-percent.yaml"
