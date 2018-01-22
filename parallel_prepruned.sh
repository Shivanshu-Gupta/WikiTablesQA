#!/bin/bash

#TODO: Synchronoize!
filename=data/WikiTableQuestions/data/training-parallel-prepruned/list.txt
while read -r line
do
  qsub -q low -N $line -o parallel_logs_prepruned/$line_$1.out -e parallel_logs_prepruned/$line_$1.err -lwalltime=12:00:00 -P cse -lselect=1:ncpus=4:mem=25gb -V -v ITERATION_NUM=$1,FILE_NAME=$line ./experiments/wikitables/scripts/prepruned/run_experiment_parallel.sh
done < $filename
