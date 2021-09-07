#!/bin/bash

echo ">>>>> Prepare all data for experiment"

echo ">>>>> Shuffle data 100 percent"
./shuffle-and-sample.sh 100

echo ">>>>> Shuffle and sample data 50 percent"
./shuffle-and-sample.sh 50
