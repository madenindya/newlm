#!/bin/bash

tasks=$1
gpu=$2

echo "Run Grid Search for BERT-Causal $tasks with GPU $gpu"

lrs=( 0.0002 0.0001 0.00005 0.00003 0.00001 0.000005 )
bss=( 32 16 )
seeds=( 1 10 42 99 386 )

for bs in "${bss[@]}"
do
    for lr in "${lrs[@]}"
    do
        for seed in "${seeds[@]}"
        do
            CUDA_VISIBLE_DEVICES=$gpu python -m newlm run_glue \
            --config_file="examples/configs_gcloud/run-100-percent.bert-causal.yaml" \
            --bs=$bs --lr=$lr --seed=$seed --tasks=$tasks
        done
    done
done

