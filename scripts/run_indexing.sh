INDEX_PATH=$1
RETRIEVER=$2
IR_DATASET=$3
CHECKPOINT=$4
mkdir -p $OUTPUT_DIR

CMD="python corpuspp.index \
    --index_path $INDEX_PATH \
    --ir_dataset $IR_DATASET \
    --retriever $RETRIEVER"

if [ -n "$CHECKPOINT" ]; then
    CMD+=" --checkpoint $CHECKPOINT"
fi

eval $CMD