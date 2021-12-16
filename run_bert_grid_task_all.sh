#!/bin/bash


echo "Run Grid Search for BERT"

./run_bert_grid_task.sh "cola" 0 &
./run_bert_grid_task.sh "mrpc" 1 &
./run_bert_grid_task.sh "rte" 2 &
./run_bert_grid_task.sh "stsb" 3 &

./run_bert_grid_task.sh "mnli" 4,5 &
./run_bert_grid_task.sh "qqp" 6,7 &

# Uncomment & Run the following after cola, mrpc, rte, stsb finished
# ./run_bert_grid_task.sh "qnli" 0,1 &
# ./run_bert_grid_task.sh "sst2" 2,3 &
