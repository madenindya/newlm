#!/bin/bash


echo "Run Grid Search for ELMO 2 Tower finetune-v1"

./run_grid-elmo-2-tower.sh "cola" 0 &
./run_grid-elmo-2-tower.sh "mrpc" 1 &
./run_grid-elmo-2-tower.sh "rte" 2 &
./run_grid-elmo-2-tower.sh "stsb" 3 &

./run_grid-elmo-2-tower.sh "mnli" 4,5 &
./run_grid-elmo-2-tower.sh "qqp" 6,7 &

# Uncomment & Run the following after cola, mrpc, rte, stsb finished
# ./run_grid-elmo-2-tower.sh "qnli" 0,1 &
# ./run_grid-elmo-2-tower.sh "sst2" 2,3 &
