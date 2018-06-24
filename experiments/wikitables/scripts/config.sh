#!/bin/bash -e

# Which fold to train on.
#FOLD=3
SCRIPT_DIR="experiments/wikitables/scripts/"
EXPERIMENT_DIR="/home/cse/dual/cs5130298/Academics/MachineComprehension/WikiTablesQA/experiments/wikitables/output/miml/experiments/$EXP_TYPE/$LF_TYPE/$EXP_NAME/FOLD_$FOLD"

# Training data.
# This is a subsample of 100 examples
TRAIN="$EXPERIMENT_DIR/../../dpd_output/random-split-$FOLD-train-$TRAIN_SIZE.examples"
# Uncomment below to use the full training set.
# TRAIN="$EXPERIMENT_DIR/random-split-$FOLD-train.examples"

# Development data used for evaluating model accuracy as training progresses.
# Using a subsample here can reduce training time.
#TRAIN_DEV="$EXPERIMENT_DIR/../../../../../examples/random-split-$FOLD-dev-100.examples"
# Uncomment below to use the full dev set.
TRAIN_DEV="$EXPERIMENT_DIR/../../../../../examples/random-split-$FOLD-dev.examples"

# Development data for evaluating the final trained model.
#DEV="$EXPERIMENT_DIR/../../../../../examples/random-split-$FOLD-dev-100.examples"
# Uncomment below to use the full dev set.
DEV="$EXPERIMENT_DIR/../../../../../examples/random-split-$FOLD-dev.examples"
# Uncomment below to use the test set.
#DEV="$EXPERIMENT_DIR/../../../../../examples/pristine-unseen-tables.examples"
TEST="$EXPERIMENT_DIR/../../../../../examples/pristine-unseen-tables.examples"

# Location of DPD output
DERIVATIONS_PATH="$EXPERIMENT_DIR/../../dpd_output/"


EPOCHS=20
MAX_TRAINING_DERIVATIONS=200
MAX_TEST_DERIVATIONS=10
BEAM_SIZE=$BEAM_SIZE
TEST_BEAM_SIZE=10
VOCAB=2

# Layer dimensionalities of semantic parser
INPUT_DIM=200
HIDDEN_DIM=100
ACTION_DIM=100
ACTION_HIDDEN_DIM=100

mkdir -p $EXPERIMENT_DIR

# echo $TRAIN
# echo $TRAIN_DEV
# echo $DEV
# echo $EXPERIMENT_NAME

# tokenfeat_ablation --encodeEntitiesWithGraph --noEntityLinkingSimilarity --tokenFeaturesOnly
# simonly_ablation --noFeatures --encodeEntitiesWithGraph
# featonly_ablation --noEntityLinkingSimilarity --encodeEntitiesWithGraph
# kg_ablation --editDistance
