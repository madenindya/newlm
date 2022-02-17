#!/bin/bash


echo "Run Grid Search for BERT-Causal"

./run_grid-causal.sh "cola" 0 &
./run_grid-causal.sh "mnli" 1 &
./run_grid-causal.sh "mrpc" 2 &
./run_grid-causal.sh "qnli" 3 &
./run_grid-causal.sh "qqp" 4 &
./run_grid-causal.sh "rte" 5 &
./run_grid-causal.sh "sst2" 6 &
./run_grid-causal.sh "stsb" 7 &
