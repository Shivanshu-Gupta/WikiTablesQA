experiments = [(K, D) for K in [-1, 15, 5, 7, 10] for D in [50, 100, 150, 200]]
with open('script.sh', 'w') as scriptf:
    for i, (K, D) in enumerate(experiments):
        scriptf.write(("qsub -P cse -N orig_all_{0} -e logs/orig_all_{0}.err -o logs/orig_all_{0}.out "
                    + "-lselect=1:highmem=1 -lwalltime=48:00:00 -V -v EXP_TYPE=orig,TRAIN_SIZE=all,K={1},D={2} "
                    + "./experiments/wikitables/scripts/run_experiment.sh\n\n").format(i, K, D))

