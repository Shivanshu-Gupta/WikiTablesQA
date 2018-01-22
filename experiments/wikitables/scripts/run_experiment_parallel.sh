#!/bin/bash -e

# change to the correct directory
cd ~/Academics/MachineComprehension/WikiTablesQA

source "experiments/wikitables/scripts/config_pruned.sh"
## Overwrite TRAIN variable, taking file name from command line argument
TRAIN="data/WikiTableQuestions/data/training-parallel/$FILE_NAME"

MY_NAME="model_pruned"
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/parser_final.ser
MODEL_DIR=$MY_DIR/models/
LFS_OUTPUT=$MY_DIR/lfs_output/

mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

echo "Training with loaded $MY_NAME model..."
./$SCRIPT_DIR/run.sh org.allenai.wikitables.WikiTablesSemanticParserCli --lfsOutput $LFS_OUTPUT --iterationNum $ITERATION_NUM --loadModel $MY_MODEL --trainingData $TRAIN --devData $TRAIN_DEV --derivationsPath $DERIVATIONS_PATH --modelOut $MY_MODEL --epochs $EPOCHS --beamSize $BEAM_SIZE --devBeamSize $TEST_BEAM_SIZE --maxDerivations $MAX_TRAINING_DERIVATIONS --vocabThreshold $VOCAB --inputDim $INPUT_DIM --hiddenDim $HIDDEN_DIM --actionDim $ACTION_DIM --actionHiddenDim $ACTION_HIDDEN_DIM --skipActionSpaceValidation --relu --actionBias --maxPoolEntityTokenSimilarities --concatLstmForDecoder --entityLinkingMlp $@ &> $MY_DIR/train_log_$FILE_NAME.txt

