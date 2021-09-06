#!/bin/bash

echo ">>>>> Run all experiments"

echo ">>>>> Create LM 100 Percent and Run Downstream GLUE"
source .env.gcloud
python -m newlm run_all --config_file="examples/configs_gcloud/run.100-percent.yaml"

echo ">>>>> Create LM 50 Percent and Run Downstream GLUE"
./shuffle-and-sample.sh 50
python -m newlm run_all --config_file="examples/configs_gcloud/run.50-percent.yaml"
