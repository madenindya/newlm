#!/bin/bash

declare -i w=$1
ens_dir=$2

maxw=$(expr $w - 1)

for i in $(seq 1 $maxw)
do
    j=$(expr $w - $i)
    echo "Run ensemble with ratio: $i $j"
    python -m newlm run_ensemble --config_file="examples/configs/run-predict-ensemble-v2.yaml" --l2r_r2l_ratio=[$i,$j]  --merge_strategy="v2" --base_dir=$ens_dir
done

tasks=( cola mnli mrpc qnli qqp rte sst2 stsb )

for task in "${tasks[@]}"
do
    python summarize_ensemble.py $ens_dir $task
done
