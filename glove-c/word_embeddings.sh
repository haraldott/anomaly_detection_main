#!/bin/bash
scriptdir="$(dirname "$0")"
cd "$scriptdir" || exit
usage() { echo "Usage: $0 [] []" 1>&2; exit 1;}

CORPUS=../data/openstack/utah/parsed/18k_spr_templates
SAVE_FILE=../data/openstack/utah/embeddings/18k_spr_templates
VECTOR_SIZE=100

while getopts c:s:v: o; do
  case $o in
    c)     CORPUS=$OPTARG;;
    s)     SAVE_FILE=$OPTARG;;
    v)     VECTOR_SIZE=$OPTARG;;
    *)     usage;;
  esac
done
shift $((OPTIND-1))

#CORPUS="$(pwd)/${CORPUS}"
#SAVE_FILE="$(pwd)/${SAVE_FILE}"
echo "$CORPUS"
echo "$SAVE_FILE"

VOCAB_FILE=vocab.txt # TODO reuse
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
MAX_ITER=1500
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
               # python2 eval/python/evaluate.py --vectors_file $SAVE_FILE.txt
               echo 'Skip eval'
           fi
       fi
    fi
  fi
fi

rm $COOCCURRENCE_FILE
rm "cooccurrence.shuf.bin"
rm "vocab.txt"
rm "${SAVE_FILE}.bin"

#run -save-file vectors -threads 8 -input-file cooccurrence.shuf.bin -x-max 10 -iter 15 -vector-size 50 -binary 2 -vocab-file vocab.txt -verbose 2
