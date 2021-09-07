#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh # change if necessary
conda activate newlm-py38
conda info

echo ">>>>> Create LM 50 Percent and Run Downstream GLUE"
source .env.gcloud
python -m newlm run_all --config_file="examples/configs_gcloud/run.50-percent.yaml"
