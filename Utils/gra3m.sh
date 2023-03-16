#!/bin/bash

export FASTA_DIR=$1
export A3M_DIR=$2
for filename in $FASTA_DIR/*.txt; do
    pid=$(basename "$filename" .txt)
    mkdir $A3M_DIR/$pid/
    amOut=$A3M_DIR/$pid/$pid.a3m
    if [ ! -f "$amOut" ]; then
        echo $filename
        hhblits -i $filename -oa3m $amOut -n 4 -d /path-To-Uniref30_2020_06/
    
    fi
done
echo "end"
