"""
Training script for Vietnamese fake news detection
Uses XLM-RoBERTa for multilingual text encoding
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import TextOnlyDataset
from models.cafe_model import TextOnlyClassifier


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vietnamese_data(train_path, test_path=None, text_col='post_message', label_col='label'):
    """Load and preprocess Vietnamese fake news dataset"""
    
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    train_df = train_df.dropna(subset=[text_col, label_col])
    
    # Clean text
    train_df[text_col] = train_df[text_col].astype(str).str.strip()
    train_df = train_df[train_df[text_col].str.len() > 10]  # Remove very short texts
    
    print(f"Training samples: {len(train_df)}")
    print(f"Label distribution:\n{train_df[label_col].value_counts()}")
    
    # Check if test file has labels
    has_test_labels = False
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if label_col in test_df.columns:
            has_test_labels = True
    
    if has_test_labels:
        print(f"Loading test data from: {test_path}")
        test_df = test_df.dropna(subset=[text_col, label_col])
        test_df[text_col] = test_df[text_col].astype(str).str.strip()
        test_df = test_df[test_df[text_col].str.len() > 10]
        # Split train into train/val
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df[label_col]
        )
    else:
        # Split training data into train/val/test
        print("Test file has no labels. Splitting training data into train/val/test.")
        train_df, test_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df[label_col]
        )
        train_df, val_df = train_test_split(
            train_df, test_size=0.125, random_state=42, stratify=train_df[label_col]
        )
    
    print(f"Final split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return {
        'train': (train_df[text_col].values, train_df[label_col].values.astype(int)),
        'val': (val_df[text_col].values, val_df[label_col].values.astype(int)),
        'test': (test_df[text_col].values, test_df[label_col].values.astype(int))
    }


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': predictions,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese Fake News Detector')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--test_path', type=str, default=None, help='Path to test CSV')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', 
                        help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_vietnamese_data(args.train_path, args.test_path)
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = TextOnlyDataset(data['train'][0], data['train'][1], tokenizer, args.max_length)
    val_dataset = TextOnlyDataset(data['val'][0], data['val'][1], tokenizer, args.max_length)
    test_dataset = TextOnlyDataset(data['test'][0], data['test'][1], tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Initialize model
    print(f"Initializing model: {args.model_name}")
    model = TextOnlyClassifier(model_name=args.model_name, num_classes=2)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_f1'].append(val_results['f1'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.4f}, Val F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with Val F1: {best_val_f1:.4f}")
    
    # Load best model and evaluate on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_results = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_results['accuracy']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_results['labels'], test_results['predictions'], 
                               target_names=['Real News', 'Fake News'], digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_results['labels'], test_results['predictions']))
    
    # Save results
    results = {
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'test_accuracy': test_results['accuracy'],
        'test_f1': test_results['f1'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'history': history
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    return results


if __name__ == '__main__':
    main()
