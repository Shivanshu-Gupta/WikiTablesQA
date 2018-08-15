#!/bin/bash -e
cd /home/cse/phd/csz178058/wikitables/elScores
source "experiments/wikitables/scripts/config.sh"

DEV=$DEV
EXT=$EXT

MY_NAME=$EXP_NAME
MY_DIR=$EXPERIMENT_DIR/$MY_NAME/
MY_MODEL=$MY_DIR/parser_final.ser
MODEL_DIR=$MY_DIR/models/

echo "Evaluating $MY_NAME development error..."
MY_DIR=$MY_DIR TEST_BEAM_SIZE=$TEST_BEAM_SIZE DEV=$DEV MAX_TEST_DERIVATIONS=$MAX_TEST_DERIVATIONS \
DERIVATIONS_PATH=$DERIVATIONS_PATH EXT=$EXT WRITE_TYPE=$WRITE_TYPE ./experiments/wikitables/scripts/eval_analysis.sh
