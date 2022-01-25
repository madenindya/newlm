#!/bin/bash


echo "Run Grid Search for BERT"

./run_grid-causal_r2l.sh "cola" 0 &
./run_grid-causal_r2l.sh "mrpc" 1 &
./run_grid-causal_r2l.sh "rte" 2 &
./run_grid-causal_r2l.sh "stsb" 3 &

./run_grid-causal_r2l.sh "mnli" 4,5 &
./run_grid-causal_r2l.sh "qqp" 6,7 &

# Uncomment & Run the following after cola, mrpc, rte, stsb finished
# ./run_grid-causal_r2l.sh "qnli" 0,1 &
# ./run_grid-causal_r2l.sh "sst2" 2,3 &
