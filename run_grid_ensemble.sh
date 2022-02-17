#!/bin/bash

declare -i w=$1
maxw=$(expr $w - 1)

for i in $(seq 1 $maxw)
do
    j=$(expr $w - $i)
    echo "Run ensemble with ratio: $i $j"
    python -m newlm run_ensemble --config_file="examples/configs/run-predict-ensemble.yaml" --l2r_r2l_ratio=[$i,$j]
done

# python -m newlm run_ensemble --config_file="examples/configs/run-predict-ensemble.yaml" --base_dir=/mnt/data1/made_workspace/newlm-output/bert-causal-en.100-percent.l2r-r2l-ensemble/checkpoint-1000000-output --l2r_r2l_ratio=[1,1]]



tasks=( cola mnli mrpc qnli qqp rte sst2 stsb )

for task in "${tasks[@]}"
do
    python summarize_ensemble.py /mnt/data1/made_workspace/newlm-output/bert-causal-en.100-percent.l2r-r2l-ensemble/checkpoint-1000000-output $task
done

#