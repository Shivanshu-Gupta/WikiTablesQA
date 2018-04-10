FOLD=5
experiments = [(K, D) for K in [-1, 15, 5, 7, 10] for D in [100]]
with open('script.sh', 'w') as scriptf:
    for K, D in experiments:
        scriptf.write(("qsub -P cse -N K{0}_D{1}_{2} -e logs/K{0}_D{1}_{2}.err -o logs/K{0}_D{1}_{2}.out "
                + "-lselect=1:ncpus=8:mem=60g -lwalltime=25:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,FOLD={2},K={0},D={1} "
                    + "./experiments/wikitables/scripts/run_experiment.sh\n\n").format(K, D, FOLD))

