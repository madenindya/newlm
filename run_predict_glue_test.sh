#!/bin/bash

# ----------- start modify

tokenizer_dir=""
    # pretrained_tokenizer from Huggingface's Hub, OR
    # folder that contains vocab.txt
    # Better use original pre-trained model dir
    #   or one of finetune checkpoint dir

model_type=""
    # Choose one of the following:
    # bert
    # bert-causal                   - for Causal L2R model
    # bert-causal-r2l               - for Causal R2L model
    # elmo-bert-causal-l2r-r2l      - for ELMO V1
    # elmo-bert-causal-l2r-r2l-v4   - for ELMO V4

best_model_dir=""
    # folder that contains cola/checkpoint-* mnli/checkpoint-* etc
    # check README for more detail structure

output_dir=""
    # use new dir

task=""
    # name of task you want to run, sepearated by comma. Ex: "cola,mrpc,rte"
    # for all task, just use ""

# ----------- end modify

# RUNING THE SCRIPT

# 0. Remove old config if exist
rm examples/gen_run.yaml

# 1. gen yaml
python script_validate_and_gen_yaml.py $tokenizer_dir $model_type $best_model_dir $output_dir $task

# 2. run predict
python -m newlm run_glue_predict \
    --config_file="examples/gen_run.yaml" \
    --test_data="test"

# 3. zip data for submission
cp $output_dir/glue-predict/*/*.tsv $output_dir/glue-predict
cp examples/glue_submission/*.tsv $output_dir/glue-predict

zip -r $output_dir/glue-predict/submission.zip $output_dir/glue-predict/*.tsv
