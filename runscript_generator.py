experiments = {
                'neither': '',
                'coverage': '--coverage', 
                'both-prob': '--coverage --templateTypeSelection probability',
                'both-mult': '--coverage --templateTypeSelection multiplier',
                'both-const-mult': '--coverage --templateTypeSelection const-multiplier',
                'sel-prob': '--templateTypeSelection probability',
                'sel-mult': '--templateTypeSelection multiplier',
                'sel-const-mult': '--templateTypeSelection const-multiplier',
                'att-loss': '--attentionLoss $type:$weight'
    }
for name, opts in experiments.items():
    with open('hpc_scripts/' + name + '.sh', 'w') as outf:
        outf.write('#!/bin/bash -e' + '\n')
        outf.write('cd ~/Academics/MachineComprehension/WikiTablesQA' + '\n')
        outf.write('MY_NAME={} FOLD=$fold SETTING=$optim$e0$ed ./experiments/wikitables/scripts/run_experiment.sh {} --optimizer $optim:e0=$e0:edecay=$ed\n'.format(name, opts))
        print('qsub -P cse -N {0} -e logs/{0}.err -o logs/{0}.out -lselect=1:ncpus=8:mem=30g -lwalltime=10:00:00 -V ./hpc_scripts/{0}.sh'.format(name))


