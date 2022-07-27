#!/bin/bash

declare -i w=$1
maxw=$(expr $w - 1)

for i in $(seq 1 $maxw)
do
    j=$(expr $w - $i)
    echo "Run ensemble with ratio: $i $j"
    python -m newlm run_ensemble --config_file="examples/configs/run-predict-ensemble-v2.yaml" --l2r_r2l_ratio=[$i,$j]  --merge_strategy="v2"
done

tasks=( cola mnli mrpc qnli qqp rte sst2 stsb )

for task in "${tasks[@]}"
do
    python summarize_ensemble.py /mnt/data4/made_workspace/newlm-output/best_l2r_r2l_finetuned/ens-l2r_4e-4-l2r_2e-4-test $task
done
