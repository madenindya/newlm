#!/bin/bash


echo "Run Grid Search for ELMO Causal L2R/R2L v4"

./run_grid-elmo-v4.sh "cola" 0 &
./run_grid-elmo-v4.sh "mrpc" 1 &
./run_grid-elmo-v4.sh "rte" 2 &
./run_grid-elmo-v4.sh "stsb" 3 &

./run_grid-elmo-v4.sh "mnli" 4,5 &
./run_grid-elmo-v4.sh "qqp" 6,7 &

# Uncomment & Run the following after cola, mrpc, rte, stsb finished
# ./run_grid-elmo-v4.sh "qnli" 0,1 &
# ./run_grid-elmo-v4.sh "sst2" 2,3 &
