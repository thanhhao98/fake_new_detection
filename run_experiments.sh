#!/bin/bash

# Run experiments for Vietnamese Fake News Detection
# This script runs both multimodal and text-only experiments

echo "==========================================="
echo "Vietnamese Fake News Detection Experiments"
echo "==========================================="

# Create output directories
mkdir -p outputs/vietnamese
mkdir -p outputs/multimodal
mkdir -p figures

# Set paths (using git submodule)
VIETNAMESE_TRAIN_PATH="./vn_fake_news/Data/train.csv"

echo ""
echo "==========================================="
echo "Step 1: Dataset Analysis"
echo "==========================================="

python analyze_dataset.py \
    --train_path "$VIETNAMESE_TRAIN_PATH" \
    --output_dir "./figures"

echo ""
echo "==========================================="
echo "Step 2: Vietnamese Text-Only Model"
echo "Using XLM-RoBERTa for Vietnamese Fake News Detection"
echo "==========================================="

python train_vietnamese.py \
    --train_path "$VIETNAMESE_TRAIN_PATH" \
    --model_name "xlm-roberta-base" \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5 \
    --max_length 256 \
    --output_dir "./outputs/vietnamese" \
    --seed 42

echo ""
echo "==========================================="
echo "Experiments Complete!"
echo "Results saved to ./outputs/"
echo "Figures saved to ./figures/"
echo "==========================================="
