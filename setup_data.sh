#!/bin/bash

echo ">>>>> Prepare all data for experiment"

echo ">>>>> 100 Percent"
./shuffle-and-sample.sh 100

echo ">>>>> 50 Percent"
./shuffle-and-sample.sh 50
