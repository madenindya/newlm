#!/bin/bash

##### ------- Start Modification

# model_*_type
#   bert
#   bert-causal
#   bert-causal-r2l

# model_*_pred_dir
#   Output prediction directory (glue-predict)
#   Ex. /home/user/bert-best/glue-predict

model_1_type="bert"
model_1_pred_dir="tmpout/bert-best-dev/glue-predict"

model_2_type="bert"
model_2_pred_dir="tmpout/bert-best-second-dev/glue-predict"

ensemble_dir="tmpout/ensnew-bert-best-first-second-dev"
    # Use new empty dir

declare -i w=100
    # sum of ensemble ratio
    # ex. w=2 --> ratio [1:1] --> weight: 0.5-0.5
    # ex. w=5 --> ratio [1:4] --> weight: 0.2-0.8
    #             ratio [2:3] --> weight: 0.4-0.6, etc
    # can try 5 first to check the result before running to 100

##### ------- End of Modification

# 1. Create structured dir
mkdir $ensemble_dir

mkdir $ensemble_dir/$model_1_type
mkdir $ensemble_dir/$model_1_type/0
cp -r $model_1_pred_dir $ensemble_dir/$model_1_type/0


if [ "$model_1_type" = "$model_2_type" ]; then
    mkdir $ensemble_dir/$model_1_type/1
    cp -r $model_2_pred_dir $ensemble_dir/$model_1_type/1
else
    mkdir $ensemble_dir/$model_2_type
    mkdir $ensemble_dir/$model_2_type/0
    cp -r $model_2_pred_dir $ensemble_dir/$model_2_type/0
fi

# 2. Create mock yaml
python script_gen_ens_yaml.py $ensemble_dir

# 3. Run ensemble
maxw=$(expr $w - 1)

for i in $(seq 1 $maxw)
do
    j=$(expr $w - $i)
    echo "Run ensemble with ratio: $i $j"
    python -m newlm run_ensemble --config_file="examples/gen_run_ens.yaml" --l2r_r2l_ratio=[$i,$j]  --merge_strategy="v2" --base_dir=$ensemble_dir
done

# 4. Summarize ensemble result
tasks=( cola mnli mrpc qnli qqp rte sst2 stsb )

for task in "${tasks[@]}"
do
    python summarize_ensemble.py $ensemble_dir $task
done

