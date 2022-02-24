#!/bin/bash

task=$1
gpu=$2

echo "Run Finetuning Grid Search for ELMO-Causal-L2R-R2L V1 $task with GPU $gpu"

lrs=( 0.0002 0.0001 0.00005 0.00003 0.00002 0.00001 0.000005 )
bss=( 32 16 )
seeds=( 1 41 386 )

for bs in "${bss[@]}"
do
    for lr in "${lrs[@]}"
    do
        for seed in "${seeds[@]}"
        do
            CUDA_VISIBLE_DEVICES=$gpu python -m newlm run_glue \
            --config_file="examples/configs_gcloud/run-ft.elmo-bert-causal-l2r-r2l-v1.yaml" \
            --bs=$bs --lr=$lr --seed=$seed --tasks=[$task]
        done
    done
done

python summarize_tuning.py outputs/en.100-percent.elmo-bert-causal-v1-finetune $task
# Warning!! output_dir is based on config file
