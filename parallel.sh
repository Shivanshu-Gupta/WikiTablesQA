#!/bin/bash

#TODO: Synchronoize!
filename=data/WikiTableQuestions/data/training-parallel/list.txt
while read -r line
do
  qsub -q low -N $line -o parallel_logs/$line.out -e parallel_logs/$line.err -lwalltime=12:00:00 -P cse -lselect=1:ncpus=4:mem=20gb -V -v ITERATION_NUM=$1,FILE_NAME=$line ./experiments/wikitables/scripts/run_experiment_parallel.sh
done < $filename

