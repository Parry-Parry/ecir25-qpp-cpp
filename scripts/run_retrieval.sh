OUTPUT_DIR='data/runs'
QUERY_PATH='data/queries.jsonl'
INDEX_PATH=$1
RETRIEVER=$2
DEPTH=${3:-1000}
CHECKPOINT=$4
mkdir -p $OUTPUT_DIR

CMD = "python corpuspp.retrieval \
    --index_path $INDEX_PATH \
    --retriever $RETRIEVER \
    --query_path $QUERY_PATH \
    --output_directory $OUTPUT_DIR \
    --depth $DEPTH"

if [ -n "$CHECKPOINT" ]; then
    CMD += " --checkpoint $CHECKPOINT"
fi

eval $CMD