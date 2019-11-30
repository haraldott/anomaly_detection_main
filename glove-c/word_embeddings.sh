#!/bin/bash

CORPUS=52k_normal_event_templates.txt
VOCAB_FILE=vocab.txt
SAVE_FILE=52k_normal_vectors

#usage() { echo "Usage: $0 [] []" 1>&2; exit 1;}
#
#while getopts ":c:v:s:" o; do
#  case "${o}" in
#    c)     CORPUS=event_templates.txt
#                ;;
#    v)      VOCAB_FILE=vocab.txt
#                ;;
#    s)      SAVE_FILE=vectors
#                ;;
#    *)
#                usage
#                ;;
#  esac
#done
#shift $((OPTIND-1))
#echo "corpus = ${CORPUS}"
#echo "vocab_file = ${VOCAB_FILE}"
#echo "save_file = ${SAVE_FILE}"

COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
VECTOR_SIZE=50
MAX_ITER=15
WINDOW_SIZE=8
BINARY=2
NUM_THREADS=8
X_MAX=10

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       if [[ $? -eq 0 ]]
       then
           if [ "$1" = 'matlab' ]; then
               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
           elif [ "$1" = 'octave' ]; then
               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
           else
               python2 eval/python/evaluate.py --vectors_file $SAVE_FILE.txt
           fi
       fi
    fi
  fi
fi

#run -save-file vectors -threads 8 -input-file cooccurrence.shuf.bin -x-max 10 -iter 15 -vector-size 50 -binary 2 -vocab-file vocab.txt -verbose 2
