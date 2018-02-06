qsub -P cse -N pos20_1 -e logs/pos20_1.err -o logs/pos20_1.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos20_1,MODEL_NAME=dev1,BEAM_SIZE=20,K=-1,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pos20_2 -e logs/pos20_2.err -o logs/pos20_2.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos20_2,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pos20_3 -e logs/pos20_3.err -o logs/pos20_3.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos20_3,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=10 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pos40_1 -e logs/pos40_1.err -o logs/pos40_1.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos40_1,MODEL_NAME=dev1,BEAM_SIZE=20,K=-1,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pos40_2 -e logs/pos40_2.err -o logs/pos40_2.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos40_2,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pos40_3 -e logs/pos40_3.err -o logs/pos40_3.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pos40_3,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=10 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pn20_1 -e logs/pn20_1.err -o logs/pn20_1.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pn20_1,MODEL_NAME=dev1,BEAM_SIZE=20,K=-1,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pn20_2 -e logs/pn20_2.err -o logs/pn20_2.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pn20_2,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pn40_1 -e logs/pn40_1.err -o logs/pn40_1.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pn40_1,MODEL_NAME=dev1,BEAM_SIZE=20,K=-1,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

qsub -P cse -N pn40_2 -e logs/pn40_2.err -o logs/pn40_2.out -lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE=pn40_2,MODEL_NAME=dev1,BEAM_SIZE=20,K=5,MARGIN=-1 ./experiments/wikitables/scripts/run_experiment.sh 

