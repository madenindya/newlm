#!/bin/bash


echo "Run Grid Search for BERT"

./run_grid-bert.sh "cola" 0 &
./run_grid-bert.sh "mrpc" 1 &
./run_grid-bert.sh "rte" 2 &
./run_grid-bert.sh "stsb" 3 &

./run_grid-bert.sh "mnli" 4,5 &
./run_grid-bert.sh "qqp" 6,7 &

# Uncomment & Run the following after cola, mrpc, rte, stsb finished
# ./run_grid-bert.sh "qnli" 0,1 &
# ./run_grid-bert.sh "sst2" 2,3 &
