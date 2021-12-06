#!/bin/bash


echo "Run Grid Search for BERT-Causal"

./run_grid_task.sh ["cola"] 0 &
./run_grid_task.sh ["mnli"] 1 &
./run_grid_task.sh ["mrpc"] 2 &
./run_grid_task.sh ["qnli"] 3 &
./run_grid_task.sh ["qqp"] 4 &
./run_grid_task.sh ["rte"] 5 &
./run_grid_task.sh ["sst2"] 6 &
./run_grid_task.sh ["stsb"] 7 &