# src/train.py

import argparse
import json
import os
import random
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# Assuming data_loader.py and model_architecture.py are in the same src directory
from .data_loader import create_data_loaders, DEFAULT_IMAGE_TRANSFORM # Or TLUFruitDataset, build_answer_vocab if needed directly
from .model_architecture import FruitVQAModel, DEFAULT_CONFIG as DEFAULT_MODEL_CONFIG

# For metrics (simple accuracy here)
from sklearn.metrics import accuracy_score

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_accuracy(logits, labels):
    """Calculates accuracy for classification tasks."""
    preds = torch.argmax(logits, dim=1)
    return accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

def save_checkpoint(state, is_best, output_dir, filename_prefix="checkpoint"):
    """Saves model checkpoint."""
    filepath = os.path.join(output_dir, f"{filename_prefix}.pth.tar")
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(output_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)
        print(f"Saved new best model to {best_filepath}")

def train_one_epoch(model, data_loader, criterion_filter, criterion_vqa,
                    optimizer, device, epoch,
                    lambda_filter=0.5, lambda_vqa=0.5, grad_clip_value=None):
    model.train()
    total_loss_filter = 0
    total_loss_vqa = 0
    total_combined_loss = 0
    
    all_filter_preds = []
    all_filter_labels = []
    all_vqa_preds = []
    all_vqa_labels = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [TRAIN]", leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        img_orig = batch['image_original'].to(device)
        img_seg = batch['image_segmented'].to(device)
        img_crop = batch['image_cropped'].to(device)
        
        q_input_ids = batch['question_input_ids'].to(device)
        q_attention_mask = batch['question_attention_mask'].to(device)
        q_token_type_ids = batch['question_token_type_ids'].to(device)
        
        filter_labels = batch['filter_label'].to(device)
        vqa_labels = batch['vqa_answer_label'].to(device)

        optimizer.zero_grad()
        
        filter_logits, vqa_logits = model(
            image_original=img_orig,
            image_segmented=img_seg,
            image_cropped=img_crop,
            question_input_ids=q_input_ids,
            question_attention_mask=q_attention_mask,
            question_token_type_ids=q_token_type_ids
        )
        
        loss_filter = criterion_filter(filter_logits, filter_labels)
        loss_vqa = criterion_vqa(vqa_logits, vqa_labels)
        
        combined_loss = (lambda_filter * loss_filter) + (lambda_vqa * loss_vqa)
        
        combined_loss.backward()
        
        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            
        optimizer.step()
        
        total_loss_filter += loss_filter.item()
        total_loss_vqa += loss_vqa.item()
        total_combined_loss += combined_loss.item()

        all_filter_preds.extend(torch.argmax(filter_logits, dim=1).cpu().numpy())
        all_filter_labels.extend(filter_labels.cpu().numpy())
        all_vqa_preds.extend(torch.argmax(vqa_logits, dim=1).cpu().numpy())
        all_vqa_labels.extend(vqa_labels.cpu().numpy())

        progress_bar.set_postfix({
            'CombLoss': f"{combined_loss.item():.4f}",
            'FiltLoss': f"{loss_filter.item():.4f}",
            'VQALoss': f"{loss_vqa.item():.4f}"
        })

    avg_loss_filter = total_loss_filter / len(data_loader)
    avg_loss_vqa = total_loss_vqa / len(data_loader)
    avg_combined_loss = total_combined_loss / len(data_loader)
    
    acc_filter = accuracy_score(all_filter_labels, all_filter_preds)
    acc_vqa = accuracy_score(all_vqa_labels, all_vqa_preds)
    
    return avg_combined_loss, avg_loss_filter, avg_loss_vqa, acc_filter, acc_vqa

def evaluate_one_epoch(model, data_loader, criterion_filter, criterion_vqa,
                       device, epoch,
                       lambda_filter=0.5, lambda_vqa=0.5):
    model.eval()
    total_loss_filter = 0
    total_loss_vqa = 0
    total_combined_loss = 0
    
    all_filter_preds = []
    all_filter_labels = []
    all_vqa_preds = []
    all_vqa_labels = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [VALID]", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            img_orig = batch['image_original'].to(device)
            img_seg = batch['image_segmented'].to(device)
            img_crop = batch['image_cropped'].to(device)
            
            q_input_ids = batch['question_input_ids'].to(device)
            q_attention_mask = batch['question_attention_mask'].to(device)
            q_token_type_ids = batch['question_token_type_ids'].to(device)
            
            filter_labels = batch['filter_label'].to(device)
            vqa_labels = batch['vqa_answer_label'].to(device)
            
            filter_logits, vqa_logits = model(
                image_original=img_orig,
                image_segmented=img_seg,
                image_cropped=img_crop,
                question_input_ids=q_input_ids,
                question_attention_mask=q_attention_mask,
                question_token_type_ids=q_token_type_ids
            )
            
            loss_filter = criterion_filter(filter_logits, filter_labels)
            loss_vqa = criterion_vqa(vqa_logits, vqa_labels)
            combined_loss = (lambda_filter * loss_filter) + (lambda_vqa * loss_vqa)
            
            total_loss_filter += loss_filter.item()
            total_loss_vqa += loss_vqa.item()
            total_combined_loss += combined_loss.item()

            all_filter_preds.extend(torch.argmax(filter_logits, dim=1).cpu().numpy())
            all_filter_labels.extend(filter_labels.cpu().numpy())
            all_vqa_preds.extend(torch.argmax(vqa_logits, dim=1).cpu().numpy())
            all_vqa_labels.extend(vqa_labels.cpu().numpy())

            progress_bar.set_postfix({
                'CombLoss': f"{combined_loss.item():.4f}",
                'FiltLoss': f"{loss_filter.item():.4f}",
                'VQALoss': f"{loss_vqa.item():.4f}"
            })

    avg_loss_filter = total_loss_filter / len(data_loader)
    avg_loss_vqa = total_loss_vqa / len(data_loader)
    avg_combined_loss = total_combined_loss / len(data_loader)
    
    acc_filter = accuracy_score(all_filter_labels, all_filter_preds)
    acc_vqa = accuracy_score(all_vqa_labels, all_vqa_preds)
    
    return avg_combined_loss, avg_loss_filter, avg_loss_vqa, acc_filter, acc_vqa

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Load Model Configuration ---
    # Start with default model config and update with any from a config file or CLI args
    model_config = DEFAULT_MODEL_CONFIG.copy()
    if args.model_config_file:
        try:
            with open(args.model_config_file, 'r') as f:
                loaded_config = json.load(f)
            model_config.update(loaded_config)
            print(f"Loaded model configuration from {args.model_config_file}")
        except Exception as e:
            print(f"Warning: Could not load model config file {args.model_config_file}: {e}. Using defaults/CLI.")

    # Override model config with specific CLI args if provided (e.g. num_vqa_answer_classes)
    # This will be updated after data loaders are created and vocab is built.

    # --- Create DataLoaders ---
    # The create_data_loaders function from data_loader.py should build the answer vocab
    # from the training set and return the number of classes.
    data_loader_config = {
        'train_annotations_path': args.train_annotations,
        'val_annotations_path': args.val_annotations,
        'test_annotations_path': args.test_annotations, # Test loader not used in this training script but good to init
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'top_k_answers': args.top_k_answers,
        'question_type_mapping': {'agricultural': 1, 'other': 0, 'unknown': 0} # Example mapping
        # Add other necessary params for create_data_loaders if any
    }
    train_loader, val_loader, _, num_vqa_classes, answer_to_idx = create_data_loaders(
        data_loader_config,
        image_transform=DEFAULT_IMAGE_TRANSFORM # Or pass a custom one
    )
    print(f"DataLoaders created. Number of VQA answer classes: {num_vqa_classes}")
    model_config['num_vqa_answer_classes'] = num_vqa_classes # CRITICAL: Update model config

    # Save answer_to_idx mapping
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "answer_to_idx.json"), 'w') as f:
        json.dump(answer_to_idx, f, indent=2)
    print(f"Saved answer_to_idx mapping to {args.output_dir}")


    # --- Initialize Model, Loss, Optimizer ---
    model = FruitVQAModel(model_config=model_config).to(device)
    print(f"FruitVQAModel initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion_filter = nn.CrossEntropyLoss()
    criterion_vqa = nn.CrossEntropyLoss()

    # Paper: AdamW optimizer (beta1=0.8, beta2=0.96)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(args.adam_beta1, args.adam_beta2),
                            weight_decay=args.weight_decay)

    # Paper has a complex LR schedule. Using ReduceLROnPlateau for simplicity.
    # For the paper's schedule, a custom LambdaLR or manual adjustment per epoch would be needed.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.lr_patience, verbose=True)

    # --- Training Loop ---
    best_val_vqa_accuracy = 0.0
    epochs_no_improve = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_combined_loss, train_filter_loss, train_vqa_loss, train_filter_acc, train_vqa_acc = train_one_epoch(
            model, train_loader, criterion_filter, criterion_vqa, optimizer, device, epoch,
            args.lambda_filter_loss, args.lambda_vqa_loss, args.grad_clip
        )
        
        val_combined_loss, val_filter_loss, val_vqa_loss, val_filter_acc, val_vqa_acc = evaluate_one_epoch(
            model, val_loader, criterion_filter, criterion_vqa, device, epoch,
            args.lambda_filter_loss, args.lambda_vqa_loss
        )

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_combined_loss:.4f} (Filt: {train_filter_loss:.4f}, VQA: {train_vqa_loss:.4f}) | "
              f"Train Acc: (Filt: {train_filter_acc:.4f}, VQA: {train_vqa_acc:.4f})")
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Valid Loss: {val_combined_loss:.4f} (Filt: {val_filter_loss:.4f}, VQA: {val_vqa_loss:.4f}) | "
              f"Valid Acc: (Filt: {val_filter_acc:.4f}, VQA: {val_vqa_acc:.4f})")

        scheduler.step(val_vqa_acc) # Step scheduler based on validation VQA accuracy

        is_best = val_vqa_acc > best_val_vqa_accuracy
        if is_best:
            best_val_vqa_accuracy = val_vqa_acc
            epochs_no_improve = 0
            print(f"New best validation VQA accuracy: {best_val_vqa_accuracy:.4f}")
        else:
            epochs_no_improve += 1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_vqa_accuracy': best_val_vqa_accuracy,
            'model_config': model_config, # Save model config with checkpoint
            'answer_to_idx': answer_to_idx # Save vocab mapping
        }, is_best, args.output_dir, filename_prefix=f"fruit_vqa_epoch_{epoch+1}")

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            break
            
    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time/3600:.2f} hours.")
    print(f"Best Validation VQA Accuracy: {best_val_vqa_accuracy:.4f}")
    print(f"Model checkpoints and logs saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fruit VQA Model")

    # Data paths
    parser.add_argument('--train_annotations', type=str, required=True, help="Path to processed training annotations JSON file.")
    parser.add_argument('--val_annotations', type=str, required=True, help="Path to processed validation annotations JSON file.")
    parser.add_argument('--test_annotations', type=str, help="Path to processed test annotations JSON file (optional, not used in training).")
    
    # Model and Training config
    parser.add_argument('--model_config_file', type=str, default=None, help="Path to a JSON file with model configurations.")
    parser.add_argument('--output_dir', type=str, default="../experiments/run01", help="Directory to save checkpoints and logs.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training.")
    
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.") # Paper mentions 50
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and validation.") # Paper mentions 32
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument('--top_k_answers', type=int, default=1000, help="Number of top answers to include in VQA vocabulary.")
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Initial learning rate for AdamW.") # Paper has specific schedule, this is a base
    parser.add_argument('--adam_beta1', type=float, default=0.8, help="AdamW beta1.") # Paper: 0.8
    parser.add_argument('--adam_beta2', type=float, default=0.96, help="AdamW beta2.") # Paper: 0.96
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument('--grad_clip', type=float, default=None, help="Value for gradient clipping (e.g., 1.0). Default: None.")
    
    parser.add_argument('--lr_patience', type=int, default=5, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Patience for early stopping. 0 to disable.")

    # Loss weights (as per paper's lambda notation, but simplified for two main tasks)
    # Paper's Eq4: L_total = lambda1*L_CE + lambda2*L_MSE + lambda3*L_PCA + L_reg
    # We use lambda_filter_loss for L_CE of filter task, lambda_vqa_loss for L_CE of VQA task
    parser.add_argument('--lambda_filter_loss', type=float, default=0.5, help="Weight for the question filter loss component.")
    parser.add_argument('--lambda_vqa_loss', type=float, default=0.5, help="Weight for the VQA answer loss component.")

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments used for this run
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
        
    main(args)