#!/bin/bash -e

# NOTE: Uncomment the following line when running on HPC
#cd /home/cse/dual/cs5130298/Academics/MachineComprehension/WikiTablesQA
source "experiments/wikitables/scripts/config_miml.sh"

MY_NAME=model_$MODEL_NAME
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/parser_final.ser
MODEL_DIR=$MY_DIR/models/
echo $MY_DIR
mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

allB=( 5   5   20  20  20  20  20)
allK=(-1   1   -1  1   5   5   5)
allM=(-1   -1  -1  -1  -1  5   10)
BEAM_SIZE=${allB[$EXP_ID]}
K=${allK[$EXP_ID]}
MARGIN=${allM[$EXP_ID]}
echo "Beam=$BEAM_SIZE K=$K Margin=$MARGIN"

echo "Training $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.WikiTablesSemanticParserCli --k $K --margin $MARGIN --trainingData $TRAIN --devData $TRAIN_DEV --derivationsPath $DERIVATIONS_PATH --modelOut $MY_MODEL --epochs $EPOCHS --beamSize $BEAM_SIZE --devBeamSize $TEST_BEAM_SIZE --maxDerivations $MAX_TRAINING_DERIVATIONS --vocabThreshold $VOCAB --inputDim $INPUT_DIM --hiddenDim $HIDDEN_DIM --actionDim $ACTION_DIM --actionHiddenDim $ACTION_HIDDEN_DIM --skipActionSpaceValidation --relu --actionBias --maxPoolEntityTokenSimilarities --concatLstmForDecoder --entityLinkingMlp $@ &> $MY_DIR/train_log.txt

# echo "Evaluating $MY_NAME training error..."
# ./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $TRAIN --model $MY_MODEL --beamSize $BEAM_SIZE --derivationsPath $DERIVATIONS_PATH --maxDerivations $MAX_TEST_DERIVATIONS &> $MY_DIR/train_error_log.txt

echo "Evaluating $MY_NAME development error..."
MY_DIR=$MY_DIR TEST_BEAM_SIZE=$TEST_BEAM_SIZE DEV=$DEV MAX_TEST_DERIVATIONS=$MAX_TEST_DERIVATIONS DERIVATIONS_PATH=$DERIVATIONS_PATH ./experiments/wikitables/scripts/eval_dev.sh
