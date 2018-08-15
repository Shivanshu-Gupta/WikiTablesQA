#!/bin/bash -e
# Usage: MY_DIR=<experiment_dir> ./experiments/wikitables/scripts/eval_dev.sh

SCRIPT_DIR="experiments/wikitables/scripts/"
MODEL_DIR=$MY_DIR/models/
MY_MODEL=$MY_DIR/parser_final.ser
DEV_LOG=$MY_DIR/dev_error_log.txt$EXT
TSV_OUT=$MY_DIR/denotations.tsv$EXT
SCORES_OUT=$MY_DIR/tokenScores$WRITE_TYPE.tsv$EXT
OFFICIAL=$MY_DIR/official_results.tsv$EXT
OFFICIAL_TXT=$MY_DIR/official_results.txt$EXT
OFFICIAL_CORRECT_MAP=$MY_DIR/official_correct_map.txt$EXT
MY_CORRECT_MAP=$MY_DIR/my_correct_map.txt$EXT
DIFF=$MY_DIR/correct_diff.txt$EXT

mkdir -p $MY_DIR
mkdir -p $MODEL_DIR

./$SCRIPT_DIR/run.sh org.allenai.wikitables.TestWikiTablesCli --testData $DEV --model $MY_MODEL --beamSize \
$TEST_BEAM_SIZE --maxDerivations $MAX_TEST_DERIVATIONS --tsvOutput $TSV_OUT --scoresOutput $SCORES_OUT \
 --writeType $WRITE_TYPE --derivationsPath $DERIVATIONS_PATH &> $DEV_LOG

#/home/cse/phd/csz178058/miniconda3/envs/python2/bin/python data/WikiTableQuestions/evaluator.py -t data/WikiTableQuestions/tagged/data/ $TSV_OUT > $OFFICIAL 2> $OFFICIAL_TXT
python2 data/WikiTableQuestions/evaluator.py -t data/WikiTableQuestions/tagged/data/ $TSV_OUT > $OFFICIAL 2> $OFFICIAL_TXT

cut -f1,2 $OFFICIAL | sort | tr '[:upper:]' '[:lower:]' | sed -e "s/[[:space:]]/ /" > $OFFICIAL_CORRECT_MAP
grep '^id: ' $DEV_LOG | sed 's/id: //' | sort > $MY_CORRECT_MAP
diff $MY_CORRECT_MAP $OFFICIAL_CORRECT_MAP > $DIFF
