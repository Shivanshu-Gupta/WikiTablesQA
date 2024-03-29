#!/bin/bash -e
# Usage: MY_DIR=<experiment_dir> ./experiments/wikitables/scripts/eval_dev.sh

# source "experiments/wikitables/scripts/config.sh"

SCRIPT_DIR="experiments/wikitables/scripts/"
MODEL_DIR=$MY_DIR/models/
MY_MODEL=$MY_DIR/parser_final.ser
DEV_LOG=$MY_DIR/test_error_log.txt
TSV_OUT=$MY_DIR/test_denotations.tsv
OFFICIAL=$MY_DIR/test_official_results.tsv
OFFICIAL_TXT=$MY_DIR/test_official_results.txt
OFFICIAL_CORRECT_MAP=$MY_DIR/test_official_correct_map.txt
MY_CORRECT_MAP=$MY_DIR/test_my_correct_map.txt
DIFF=$MY_DIR/test_correct_diff.txt

mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $DEV --model $MY_MODEL --beamSize $TEST_BEAM_SIZE --maxDerivations $MAX_TEST_DERIVATIONS --tsvOutput $TSV_OUT --derivationsPath $DERIVATIONS_PATH --randomSeed $SEED &> $DEV_LOG

python data/WikiTableQuestions/evaluator.py -t data/WikiTableQuestions/tagged/data/ $TSV_OUT > $OFFICIAL 2> $OFFICIAL_TXT

cut -f1,2 $OFFICIAL | sort | tr '[:upper:]' '[:lower:]' | sed -e "s/[[:space:]]/ /" > $OFFICIAL_CORRECT_MAP
grep '^id: ' $DEV_LOG | sed 's/id: //' | sort > $MY_CORRECT_MAP
diff $MY_CORRECT_MAP $OFFICIAL_CORRECT_MAP > $DIFF
