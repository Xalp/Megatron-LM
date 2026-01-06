#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

CHECKPOINT_PATH=${1:-"checkpoints/qwen3_06b"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/qwen3_06b"}
TOKENIZER_ARG=${3:-"Qwen/Qwen3-0.6B"} # Path to tokenizer model, defaults to HF hub path
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Create directories if they don't exist
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
NUM_LAYERS=28
DTYPE="bf16"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=32768 # Matching config.json

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="${PWD}/benchmark_cache_qwen3_06b"
mkdir -p "$DATA_CACHE_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --qk-layernorm
    --init-method-std 0.02
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 1953125000 # Tuned for demonstration, adjust as needed
    --lr-decay-samples 1949218748
    --lr-warmup-samples 3906252
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE
    --sequence-parallel
)

# Distributed Data Parallel (DDP) arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$DATA_ARG" == "MOCK" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type HuggingFaceTokenizer"
        "--tokenizer-model $TOKENIZER_ARG"
        "--vocab-size 151936"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer"
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        "--vocab-size 151936"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 32
    --eval-interval 100
    --save-interval 1000
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
