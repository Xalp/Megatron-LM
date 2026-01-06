# Qwen3 0.6B Training Guide

This directory contains resources to train the Qwen3 0.6B model using Megatron-LM.

## 1. Environment Setup

Ensure you have the following installed:
- PyTorch with CUDA support
- Megatron-LM dependencies (see root README)
- `transformers` library (compatible with Qwen3)
- `accelerate` (optional, but good practice)

## 2. Data Preparation

Megatron-LM requires data to be preprocessed into binary format (`.bin` and `.idx`) before training.

### Step 2.1: Prepare your data
Format your training data as a JSON file, where each line is a JSON object with a text field (usually `text` or `content`).
Example `data.json`:
```json
{"text": "This is the first document."}
{"text": "This is another document."}
```

### Step 2.2: Preprocess the data
Use the provided helper script `preprocess_qwen3.sh` to tokenize your data.

```bash
# Usage: ./preprocess_qwen3.sh <INPUT_JSON> <OUTPUT_PREFIX> <TOKENIZER_PATH>
# Example:
chmod +x examples/qwen3/06btest/preprocess_qwen3.sh
./examples/qwen3/06btest/preprocess_qwen3.sh \
    /pfs/ziqijin/Megatron-LM/examples/qwen3/06btest/codeparrot_data.json \
    examples/qwen3/06btest/data/my_qwen_data \
    Qwen/Qwen3-0.6B
```

This will generate `examples/qwen3/06btest/data/my_qwen_data_text_document.bin` and `.idx`.

## 3. Training

Start training using the `pretrain_qwen3_06b.sh` script.

```bash
# Usage: ./pretrain_qwen3_06b.sh <CHECKPOINT_PATH> <TENSORBOARD_PATH> <TOKENIZER_PATH> <DATA_PREFIX>

# Example:
chmod +x examples/qwen3/06btest/pretrain_qwen3_06b.sh

./examples/qwen3/06btest/pretrain_qwen3_06b.sh \
    checkpoints/qwen3_06b \
    tensorboard_logs/qwen3_06b \
    Qwen/Qwen3-0.6B \
    examples/qwen3/06btest/data/my_qwen_data_text_document
```

**Note**: The `<DATA_PREFIX>` should be the path to the `.bin` file **without** the extension. If your file is `my_data_text_document.bin`, use `my_data_text_document`.

## 4. Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/qwen3_06b
```

## 5. Converting to HuggingFace Format

To convert the Megatron-LM checkpoint back to HuggingFace format for inference, you can use the conversion scripts provided in `examples/qwen3/` or standard Megatron conversion tools (check `tools/checkpoint/convert.py` or similar if available for Qwen).
