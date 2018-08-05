#!/bin/bash -e

SCRIPT_DIR="experiments/wikitables/scripts/"
# Which fold to train on.
# FOLD=1

# Training data.
# This is a subsample of 100 examples
# TRAIN="data/WikiTableQuestions/data/subsamples/random-split-$FOLD-train-100.examples"
# Uncomment below to use the full training set.
#TRAIN="data/WikiTableQuestions/data/random-split-$FOLD-train.examples"
# TRAIN="data/WikiTableQuestions/data/random-split-$FOLD-train-dpd.examples"
#TRAIN="data/WikiTableQuestions/data/annotated-all.examples"
# TRAIN="data/WikiTableQuestions/data/annotated-train.examples"
#TRAIN="data/WikiTableQuestions/data/training.examples"

# Development data used for evaluating model accuracy as training progresses.
# Using a subsample here can reduce training time.
# TRAIN_DEV="data/WikiTableQuestions/data/subsamples/random-split-$FOLD-dev-100.examples"
# Uncomment below to use the full dev set.
#TRAIN_DEV="data/WikiTableQuestions/data/random-split-$FOLD-dev.examples"
# TRAIN_DEV="data/WikiTableQuestions/data/annotated-dev.examples"

# DEV="data/WikiTableQuestions/data/random-split-$FOLD-train.examples"
# Development data for evaluating the final trained model.
# DEV="data/WikiTableQuestions/data/subsamples/random-split-$FOLD-dev-100.examples"
# Uncomment below to use the full dev set.
#DEV="data/WikiTableQuestions/data/random-split-$FOLD-dev.examples"
# Uncomment below to use the test set.
# DEV="data/WikiTableQuestions/data/pristine-unseen-tables.examples"
# DEV="data/WikiTableQuestions/data/annotated-dev.examples"

# Location of DPD output
DERIVATIONS_PATH="data/dpd_output/$DPD/"

EXPERIMENT_NAME="fold$FOLD"
EXPERIMENT_ID="00"
EXPERIMENT_DIR="experiments/wikitables/output/$METHOD/$EXPERIMENT_ID/$EXPERIMENT_NAME/"

EPOCHS=20
MAX_TRAINING_DERIVATIONS=100
MAX_TEST_DERIVATIONS=10
BEAM_SIZE=5
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
