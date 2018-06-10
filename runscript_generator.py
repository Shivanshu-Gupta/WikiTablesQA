experiments = {
                'neither':      ('', []),
                'coverage':     ('--coverage', []),
                'both-prob':    ('--coverage --templateTypeSelection probability', []),
                'both-mult':    ('--coverage --templateTypeSelection multiplier', []),
                'both-const-mult': ('--coverage --templateTypeSelection const-multiplier', []),
                'sel-prob':     ('--templateTypeSelection probability', []),
                'sel-mult':     ('--templateTypeSelection multiplier', []),
                'sel-const-mult': ('--templateTypeSelection const-multiplier', []),
                'att-loss':     ('--attentionLoss $type:$weight', ['type', 'weight']),
                'tree':         ('--useParentInput $parent', ['parent']),
                'no-prev':      ('--useParentInput $parent --useParentRnnState', ['parent']),
                'pos':          ('--useHolePosition --holePositionDim $hole', ['hole'])
    }
optim_params = ['optim', 'e0', 'ed']

def _dump_runscript(name, hyperparams, opts):
    with open('hpc_scripts/' + name + '.sh', 'w') as outf:
        outf.write('#!/bin/bash -e' + '\n')
        outf.write('cd ~/Academics/MachineComprehension/WikiTablesQA' + '\n')
        outf.write('MY_NAME={} FOLD=$fold SETTING={} ./experiments/wikitables/scripts/run_experiment.sh {} --optimizer ${{optim}}:e0=${{e0}}:edecay=${{ed}}'.format(name, hyperparams, opts) + '\n')
        print('qsub -P cse -N {0} -e logs/{0}.err -o logs/{0}.out -lselect=1:ncpus=8:mem=30g -lwalltime=10:00:00 -V ./hpc_scripts/{0}.sh'.format(name))

def _hyperparams(optim_params, model_params):
    return ''.join(['${' + param + '}' for param in model_params + optim_params])

def generate_combination(exps):
    name = '-'.join(exps)
    model_params = [param for exp in exps for param in experiments[exp][1]]
    hyperparams = _hyperparams(optim_params, model_params)
    opts = ' '.join([experiments[exp][0] for exp in exps])
    _dump_runscript(name, hyperparams, opts)

if __name__ == '__main__':
    # dump all individual experiments
    for name, (opts, model_params) in experiments.items():
        hyperparams = _hyperparams(optim_params, model_params)
        _dump_runscript(name, hyperparams, opts)

