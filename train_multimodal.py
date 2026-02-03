"""
Training script for multimodal fake news detection using CAFE model
Based on Twitter/Weibo datasets with pre-extracted features
"""

import os
import sys
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import FeatureDataset
from models.cafe_model import SimilarityModule, CAFEModel


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_contrastive_data(text, image, label):
    """Prepare matched/unmatched pairs for contrastive learning"""
    real_index = [i for i, l in enumerate(label) if l == 0]
    if len(real_index) < 2:
        real_index = list(range(min(2, len(label))))
    
    text_real = text[real_index]
    image_real = image[real_index]
    matched_image = copy.deepcopy(image_real)
    unmatched_image = copy.deepcopy(image_real).roll(shifts=3, dims=0)
    
    return text_real, matched_image, unmatched_image


def train_epoch(similarity_module, detection_module, train_loader, 
                optim_similarity, optim_detection, 
                loss_func_similarity, loss_func_detection, device):
    """Train for one epoch"""
    
    similarity_module.train()
    detection_module.train()
    
    total_similarity_loss = 0
    total_detection_loss = 0
    predictions = []
    labels = []
    
    for text, image, label in tqdm(train_loader, desc="Training"):
        text = text.to(device)
        image = image.to(device)
        label = label.to(device)
        
        # Prepare contrastive pairs
        fixed_text, matched_image, unmatched_image = prepare_contrastive_data(text, image, label)
        fixed_text = fixed_text.to(device)
        matched_image = matched_image.to(device)
        unmatched_image = unmatched_image.to(device)
        
        # Task 1: Similarity learning
        text_aligned_match, image_aligned_match, _ = similarity_module(fixed_text, matched_image)
        text_aligned_unmatch, image_aligned_unmatch, _ = similarity_module(fixed_text, unmatched_image)
        
        similarity_label = torch.cat([
            torch.ones(text_aligned_match.shape[0]),
            -torch.ones(text_aligned_unmatch.shape[0])
        ], dim=0).to(device)
        
        text_aligned_all = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
        image_aligned_all = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
        
        loss_similarity = loss_func_similarity(text_aligned_all, image_aligned_all, similarity_label)
        
        optim_similarity.zero_grad()
        loss_similarity.backward()
        optim_similarity.step()
        
        # Task 2: Detection
        text_aligned, image_aligned, _ = similarity_module(text, image)
        pred_detection = detection_module(text, image, text_aligned.detach(), image_aligned.detach())
        loss_detection = loss_func_detection(pred_detection, label)
        
        optim_detection.zero_grad()
        loss_detection.backward()
        optim_detection.step()
        
        total_similarity_loss += loss_similarity.item()
        total_detection_loss += loss_detection.item()
        
        preds = torch.argmax(pred_detection, dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(label.cpu().numpy())
    
    avg_sim_loss = total_similarity_loss / len(train_loader)
    avg_det_loss = total_detection_loss / len(train_loader)
    accuracy = accuracy_score(labels, predictions)
    
    return avg_sim_loss, avg_det_loss, accuracy


def evaluate(similarity_module, detection_module, test_loader, 
             loss_func_detection, device):
    """Evaluate the model"""
    
    similarity_module.eval()
    detection_module.eval()
    
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for text, image, label in tqdm(test_loader, desc="Evaluating"):
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)
            
            text_aligned, image_aligned, _ = similarity_module(text, image)
            pred_detection = detection_module(text, image, text_aligned, image_aligned)
            loss = loss_func_detection(pred_detection, label)
            
            total_loss += loss.item()
            preds = torch.argmax(pred_detection, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(label.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Fake News Detector')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='./outputs_multimodal')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_set = FeatureDataset(
        os.path.join(args.data_dir, 'train_text_with_label.npz'),
        os.path.join(args.data_dir, 'train_image_with_label.npz')
    )
    test_set = FeatureDataset(
        os.path.join(args.data_dir, 'test_text_with_label.npz'),
        os.path.join(args.data_dir, 'test_image_with_label.npz')
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")
    
    # Initialize models
    similarity_module = SimilarityModule()
    similarity_module.to(device)
    
    detection_module = CAFEModel()
    detection_module.to(device)
    
    # Loss functions
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    
    # Optimizers
    optim_similarity = torch.optim.Adam(similarity_module.parameters(), lr=args.lr)
    optim_detection = torch.optim.Adam(detection_module.parameters(), lr=args.lr)
    
    # Training loop
    best_acc = 0
    history = {'sim_loss': [], 'det_loss': [], 'train_acc': [], 'test_acc': [], 'test_f1': []}
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        sim_loss, det_loss, train_acc = train_epoch(
            similarity_module, detection_module, train_loader,
            optim_similarity, optim_detection,
            loss_func_similarity, loss_func_detection, device
        )
        
        test_results = evaluate(
            similarity_module, detection_module, test_loader,
            loss_func_detection, device
        )
        
        history['sim_loss'].append(sim_loss)
        history['det_loss'].append(det_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_results['accuracy'])
        history['test_f1'].append(test_results['f1'])
        
        print(f"Sim Loss: {sim_loss:.4f}, Det Loss: {det_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_results['accuracy']:.4f}, Test F1: {test_results['f1']:.4f}")
        
        if test_results['accuracy'] > best_acc:
            best_acc = test_results['accuracy']
            torch.save({
                'similarity_module': similarity_module.state_dict(),
                'detection_module': detection_module.state_dict()
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with Test Acc: {best_acc:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    similarity_module.load_state_dict(checkpoint['similarity_module'])
    detection_module.load_state_dict(checkpoint['detection_module'])
    
    test_results = evaluate(
        similarity_module, detection_module, test_loader,
        loss_func_detection, device
    )
    
    print(f"\nTest Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1: {test_results['f1']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_results['labels'], test_results['predictions'],
                               target_names=['Real News', 'Fake News'], digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_results['labels'], test_results['predictions']))
    
    # Save results
    results = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'best_test_accuracy': best_acc,
        'final_test_accuracy': test_results['accuracy'],
        'final_test_f1': test_results['f1'],
        'history': history
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    main()
