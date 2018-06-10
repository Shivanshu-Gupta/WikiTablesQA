
# neither/coverage/sel-prob experiments run script
model=coverage optim=sgd e0s=(0.1) eds=(0.01) folds=(1 2 3 4 5)
for fold in $folds
for e0 in $e0s
for ed in $eds
qsub -q low -P cse -N $fold$e0$ed$model$optim -e logs/$model$fold$optim$e0$ed.err -o logs/$model$fold$optim$e0$ed.out -lselect=1:ncpus=8:mem=35g -lwalltime=14:00:00 -v optim=$optim,e0=$e0,ed=$ed,fold=$fold -V ./hpc_scripts/$model.sh

# attention loss experiments run script
model=att-loss optim=sgd e0s=(0.1) eds=(0.01) folds=(1)
types=(all entityTemplates) weights=(0.01 0.001 0.0001 0.00001 0.000001)
for type in $types
for weight in $weights
for fold in $folds
for e0 in $e0s
for ed in $eds
qsub -q low -P cse -N $weight$type$model -e logs/$model$type$weight$fold$optim$e0$ed.err -o logs/$model$type$weight$fold$optim$e0$ed.out -lselect=1:ncpus=4:mem=30g -lwalltime=14:00:00 -v optim=$optim,e0=$e0,ed=$ed,fold=$fold,type=$type,weight=$weight -V ./hpc_scripts/$model.sh
