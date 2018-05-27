Ks=(-1 15 5 7 10) Ds=(100) folds=(1 2 3 4 5)
for fold in $folds
for K in $Ks
for D in $Ds
qsub -P cse -N ${fold}K${K}D${D} -e logs/${fold}K${K}D${D}.err -o logs/${fold}K${K}D${D}.out -lselect=1:ncpus=8:mem=60g -lwalltime=25:00:00 -V -v EXP_TYPE=orig,LF_TYPE=prepruned,TRAIN_SIZE=all,FOLD=$fold,K=$K,D=$D ./experiments/wikitables/scripts/run_experiment.sh
