#!/bin/bash -e

EXP_NAME=K${K}

# NOTE: Uncomment the following line when running on HPC
cd /home/cse/dual/cs5130298/Academics/MachineComprehension/WikiTablesQA
source "experiments/wikitables/scripts/config.sh"
MAX_TRAINING_DERIVATIONS=$D

MY_NAME=model_${TRAIN_SIZE}_D${D}
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/parser_final.ser
MODEL_DIR=$MY_DIR/models/
echo $MY_DIR
mkdir -p $MY_DIR
mkdir -p $MODEL_DIR
echo $TRAIN

SEED=2732932987
echo "Training $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.WikiTablesSemanticParserCli --k $K --margin -1 --trainingData $TRAIN --devData $TRAIN_DEV --derivationsPath $DERIVATIONS_PATH --modelOut $MY_MODEL --epochs $EPOCHS --beamSize -1 --devBeamSize $TEST_BEAM_SIZE --maxDerivations $MAX_TRAINING_DERIVATIONS --vocabThreshold $VOCAB --inputDim $INPUT_DIM --hiddenDim $HIDDEN_DIM --actionDim $ACTION_DIM --actionHiddenDim $ACTION_HIDDEN_DIM --skipActionSpaceValidation --relu --actionBias --maxPoolEntityTokenSimilarities --concatLstmForDecoder --entityLinkingMlp --randomSeed $SEED $@ &> $MY_DIR/train_log.txt

# echo "Evaluating $MY_NAME training error..."
# ./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $TRAIN --model $MY_MODEL --beamSize $BEAM_SIZE --derivationsPath $DERIVATIONS_PATH --maxDerivations $MAX_TEST_DERIVATIONS &> $MY_DIR/train_error_log.txt

echo "Evaluating $MY_NAME development error..."
SEED=$SEED MY_DIR=$MY_DIR TEST_BEAM_SIZE=$TEST_BEAM_SIZE DEV=$DEV MAX_TEST_DERIVATIONS=$MAX_TEST_DERIVATIONS DERIVATIONS_PATH=$DERIVATIONS_PATH ./experiments/wikitables/scripts/eval_dev.sh

echo "Evaluating $MY_NAME test error..."
SEED=$SEED MY_DIR=$MY_DIR TEST_BEAM_SIZE=$TEST_BEAM_SIZE DEV=$TEST MAX_TEST_DERIVATIONS=$MAX_TEST_DERIVATIONS DERIVATIONS_PATH=$DERIVATIONS_PATH ./experiments/wikitables/scripts/eval_test.sh
