#!/bin/bash

# echo "========== CREATE THE DATA =========="
# ./shuffle-and-sample.sh 1

# echo "========== RUN ALL (BUILD LM + BENCHMARK GLUE) ========== "
# source .env.template
# python -m newlm run_all --config_file="examples/configs/run.1-percent.yaml"

echo "========== Continue Run Glue ========== "
source .env.template
python -m newlm run_glue --config_file="examples/configs/run.1-percent.glue.yaml"
