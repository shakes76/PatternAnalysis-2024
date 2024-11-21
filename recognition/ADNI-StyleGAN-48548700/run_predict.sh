#!/bin/bash

# Define the range of epochs
start_epoch=0
end_epoch=150
increment=5

for epoch in $(seq $start_epoch $increment $end_epoch)
do
    echo "Running predict.py for epoch $epoch"
    python predict.py --epoch=$epoch
done

