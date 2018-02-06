experiments = [("pos20_1",  20, -1,  -1),
              ("pos20_2",   20,  5,  -1),
              ("pos20_3",   20,  5,  10),
              ("pos40_1",   20, -1,  -1),
              ("pos40_2",   20,  5,  -1),
              ("pos40_3",   20,  5,  10),
              ("pn20_1",    20, -1,  -1),
              ("pn20_2",    20,  5,  -1),
              ("pn40_1",    20, -1,  -1),
              ("pn40_2",    20,  5,  -1)]
with open('script.sh', 'w') as scriptf:
    for name, beam, k, margin in experiments:
        scriptf.write(("qsub -P cse -N {0} -e logs/{0}.err -o logs/{0}.out "
            + "-lselect=1:ncpus=8:ngpus=1:mem=20G -lwalltime=48:00:00 -V -v EXPERIMENT_TYPE={0},"
            + "MODEL_NAME=dev1,BEAM_SIZE={1},K={2},MARGIN={3} "
            + "./experiments/wikitables/scripts/run_experiment.sh \n\n").format(name, beam, k, margin))

