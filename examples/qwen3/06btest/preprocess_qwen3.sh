#!/bin/bash

# Configuration
# You should update these paths to where your data and tokenizer are located
INPUT_DATA=${1:-"data/qwen_sample_data.json"}  # Input JSON data
OUTPUT_PREFIX=${2:-"data/qwen3_06b_tokenized"} # Output prefix for .bin and .idx files
TOKENIZER_MODEL=${3:-"Qwen/Qwen3-0.6B"}       # HuggingFace model path or local directory

# Create output directory if it doesn't exist
mkdir -p $(dirname "$OUTPUT_PREFIX")

# Run preprocessing
# Using HuggingFaceTokenizer which handles Qwen's BPE 
# Ensure you have 'transformers>=4.32.0' (or appropriate version for Qwen) installed

python tools/preprocess_data.py \
       --input "$INPUT_DATA" \
       --output-prefix "$OUTPUT_PREFIX" \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model "$TOKENIZER_MODEL" \
       --append-eod \
       --workers 64
