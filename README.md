# Vietnamese Fake News Detection

A multimodal fake news detection system based on the CAFE (Cross-modal Ambiguity Learning) framework, adapted for Vietnamese text classification.

## Overview

This project implements fake news detection using:
- **CAFE Model**: Cross-modal Ambiguity Learning for Multimodal Fake News Detection (WWW 2022)
- **XLM-RoBERTa**: Multilingual transformer for Vietnamese text encoding
- **Vietnamese Dataset**: ReINTEL dataset from VLSP 2020 challenge

### Results

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| XLM-RoBERTa Base | **90.04%** | **0.8969** | 0.8959 | 0.9004 |

**Per-Class Performance**:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Real News | 92.39% | 95.70% | 94.01% |
| Fake News | 77.07% | 64.71% | 70.35% |

## Project Structure

```
fake_news_detection/
├── vn_fake_news/            # Git submodule: Vietnamese dataset
│   └── Data/
│       └── train.csv        # Training data (5,171 samples)
├── figures/                 # Generated analysis figures
├── models/
│   └── cafe_model.py        # CAFE model implementation
├── utils/
│   └── dataset.py           # Dataset utilities
├── outputs/                 # Experiment outputs
│   └── vietnamese/
│       ├── best_model.pt    # Best model checkpoint
│       └── results.json     # Training results
├── analyze_dataset.py       # Dataset analysis script
├── train_vietnamese.py      # Vietnamese training script
├── train_multimodal.py      # Multimodal training script
├── run_experiments.sh       # Experiment runner
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/YOUR_USERNAME/fake_news_detection.git
cd fake_news_detection
```

Or if you've already cloned:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- torch >= 1.9.0
- transformers >= 4.20.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- tqdm >= 4.62.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## Usage

### Quick Start

Run all experiments with a single command:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### Dataset Analysis

Generate visualizations and statistics for the Vietnamese dataset:

```bash
python analyze_dataset.py \
    --train_path "./vn_fake_news/Data/train.csv" \
    --output_dir "./figures"
```

**Generated figures**:
- `dataset_overview.png` - Combined overview
- `label_distribution.png` - Class distribution
- `text_length_analysis.png` - Text length statistics
- `word_frequency.png` - Top words analysis
- `engagement_metrics.png` - Likes, shares, comments
- `summary_statistics.png` - Summary table

### Train Vietnamese Model

Train the XLM-RoBERTa model on Vietnamese fake news data:

```bash
python train_vietnamese.py \
    --train_path "./vn_fake_news/Data/train.csv" \
    --model_name "xlm-roberta-base" \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 256 \
    --output_dir "./outputs/vietnamese"
```

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--train_path` | Required | Path to training CSV |
| `--model_name` | `xlm-roberta-base` | Pretrained model name |
| `--batch_size` | 16 | Training batch size |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max token length |
| `--output_dir` | `./outputs` | Output directory |
| `--seed` | 42 | Random seed |

### Train Multimodal Model (CAFE)

For multimodal training with pre-extracted features:

```bash
python train_multimodal.py \
    --data_dir "./data/twitter" \
    --epochs 100 \
    --batch_size 64
```

**Note**: Requires pre-extracted text and image features in `.npz` format.

## Dataset

### Vietnamese Fake News Dataset (ReINTEL)

The dataset is included as a git submodule from:
- **Source**: [PoLsss/Read_time_Vietnamese_FakeNews_Detection](https://github.com/PoLsss/Read_time_Vietnamese_FakeNews_Detection)
- **Original**: VLSP 2020 ReINTEL Challenge

**Statistics**:
| Metric | Value |
|--------|-------|
| Total Samples | 5,171 |
| Real News | 4,237 (82%) |
| Fake News | 934 (18%) |
| Avg. Text Length | 770 characters |
| Avg. Word Count | 140 words |

**Data Fields**:
- `post_message`: Vietnamese text content
- `label`: 0 (real) or 1 (fake)
- `num_like_post`: Number of likes
- `num_share_post`: Number of shares
- `num_comment_post`: Number of comments

## Model Architecture

### Text-Only Model (Vietnamese)

```
Input Text
    ↓
XLM-RoBERTa Encoder (768-dim)
    ↓
Projection Layer (768 → 256)
    ↓
BatchNorm + ReLU + Dropout
    ↓
Classification Head (256 → 2)
    ↓
Softmax → [Real, Fake]
```

### CAFE Model (Multimodal)

```
Text Input          Image Input
    ↓                   ↓
FastCNN (200→128)   VGG-19 (512→128)
    ↓                   ↓
    └───────┬───────────┘
            ↓
    Similarity Module
            ↓
    Ambiguity Learning (KL Divergence)
            ↓
    Weighted Fusion
            ↓
    Classification → [Real, Fake]
```

## References

1. Chen, Y., et al. (2022). **Cross-modal Ambiguity Learning for Multimodal Fake News Detection**. WWW 2022.
   - Paper: https://dl.acm.org/doi/10.1145/3485447.3512042
   - Code: https://github.com/cyxanna/CAFE

2. VLSP 2020. **ReINTEL: Reliable Intelligence Identification on Vietnamese SNS**.
   - Official: https://vlsp.org.vn/vlsp2020/eval/reintel

3. Conneau, A., et al. (2020). **Unsupervised Cross-lingual Representation Learning at Scale**.
   - Model: https://huggingface.co/xlm-roberta-base

## License

This project is for educational and research purposes.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{chen2022cafe,
  title={Cross-modal Ambiguity Learning for Multimodal Fake News Detection},
  author={Chen, Yixuan and Li, Dongsheng and Zhang, Peng and others},
  booktitle={Proceedings of The Web Conference 2022},
  pages={2897--2906},
  year={2022}
}
```
