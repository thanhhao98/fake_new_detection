"""
Dataset utilities for fake news detection
Supports both Twitter/MediaEval and Vietnamese fake news datasets
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import os


class TextOnlyDataset(Dataset):
    """Dataset for text-only fake news detection"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features (text + image)"""
    
    def __init__(self, text_file, image_file=None):
        text_data = np.load(text_file)
        self.text_features = torch.from_numpy(text_data["data"]).float()
        self.labels = torch.from_numpy(text_data["label"]).long()
        
        if image_file and os.path.exists(image_file):
            image_data = np.load(image_file)
            self.image_features = torch.from_numpy(image_data["data"]).squeeze().float()
        else:
            # Create dummy image features if not available
            self.image_features = torch.zeros(len(self.labels), 512)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.image_features[idx], self.labels[idx]


def load_vietnamese_dataset(train_path, test_path=None, text_col='post_message', label_col='label'):
    """Load Vietnamese fake news dataset from CSV files"""
    
    train_df = pd.read_csv(train_path)
    train_df = train_df.dropna(subset=[text_col, label_col])
    
    train_texts = train_df[text_col].values
    train_labels = train_df[label_col].values.astype(int)
    
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_df = test_df.dropna(subset=[text_col, label_col])
        test_texts = test_df[text_col].values
        test_labels = test_df[label_col].values.astype(int)
    else:
        # Split train into train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
    
    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }


def create_dataloaders(dataset, batch_size=32, num_workers=4):
    """Create train/val/test dataloaders"""
    
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_tokenizer(model_name='xlm-roberta-base'):
    """Get tokenizer for the specified model"""
    return AutoTokenizer.from_pretrained(model_name)
