qsub -P cse -N orig_all_0 -e logs/orig_all_0.err -o logs/orig_all_0.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=-1,D=50 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_1 -e logs/orig_all_1.err -o logs/orig_all_1.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=-1,D=100 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_2 -e logs/orig_all_2.err -o logs/orig_all_2.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=-1,D=150 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_3 -e logs/orig_all_3.err -o logs/orig_all_3.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=-1,D=200 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_4 -e logs/orig_all_4.err -o logs/orig_all_4.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=15,D=50 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_5 -e logs/orig_all_5.err -o logs/orig_all_5.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=15,D=100 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_6 -e logs/orig_all_6.err -o logs/orig_all_6.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=15,D=150 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_7 -e logs/orig_all_7.err -o logs/orig_all_7.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=15,D=200 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_8 -e logs/orig_all_8.err -o logs/orig_all_8.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=5,D=50 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_9 -e logs/orig_all_9.err -o logs/orig_all_9.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=5,D=100 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_10 -e logs/orig_all_10.err -o logs/orig_all_10.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=5,D=150 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_11 -e logs/orig_all_11.err -o logs/orig_all_11.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=5,D=200 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_12 -e logs/orig_all_12.err -o logs/orig_all_12.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=7,D=50 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_13 -e logs/orig_all_13.err -o logs/orig_all_13.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=7,D=100 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_14 -e logs/orig_all_14.err -o logs/orig_all_14.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=7,D=150 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_15 -e logs/orig_all_15.err -o logs/orig_all_15.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=7,D=200 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_16 -e logs/orig_all_16.err -o logs/orig_all_16.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=10,D=50 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_17 -e logs/orig_all_17.err -o logs/orig_all_17.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=10,D=100 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_18 -e logs/orig_all_18.err -o logs/orig_all_18.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=10,D=150 ./experiments/wikitables/scripts/run_experiment.sh

qsub -P cse -N orig_all_19 -e logs/orig_all_19.err -o logs/orig_all_19.out -lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K=10,D=200 ./experiments/wikitables/scripts/run_experiment.sh

