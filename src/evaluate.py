# src/evaluate.py

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

# Assuming data_loader.py and model_architecture.py are in the same src directory
from .data_loader import TLUFruitDataset, DEFAULT_IMAGE_TRANSFORM, UNK_TOKEN, PAD_TOKEN
from .model_architecture import FruitVQAModel # DEFAULT_CONFIG is also available if needed

from sklearn.metrics import accuracy_score, f1_score, classification_report

# For WUPS score
import nltk
try:
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
except ImportError:
    print("NLTK not found. Please install it: pip install nltk")
    wordnet = None
    WordNetLemmatizer = None

def download_nltk_data_if_needed():
    """Downloads NLTK WordNet if not already present."""
    if wordnet is None: # NLTK itself not imported
        return False
    try:
        wordnet.ensure_loaded()
        print("WordNet is already available.")
        return True
    except LookupError:
        print("WordNet data not found. Attempting to download...")
        try:
            nltk.download('wordnet')
            nltk.download('omw-1.4') # Open Multilingual WordNet
            # Re-check after download
            wordnet.ensure_loaded()
            print("WordNet downloaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to download WordNet: {e}")
            print("Please download WordNet manually: import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')")
            return False

# Initialize lemmatizer globally if NLTK is available
LEMMA = None
if WordNetLemmatizer:
    LEMMA = WordNetLemmatizer()

def get_wordnet_synset(word):
    """Get the first noun synset for a word, lemmatizing first."""
    if not wordnet or not LEMMA:
        return None
    if not word:
        return None
    lemma = LEMMA.lemmatize(word.lower(), pos='n') # Lemmatize as noun
    synsets = wordnet.synsets(lemma, pos=wordnet.NOUN)
    return synsets[0] if synsets else None

def calculate_wups(word1, word2, similarity_threshold=0.9):
    """
    Calculates Wu-Palmer Similarity (WUPS) between two words.
    Returns the WUPS score and a boolean indicating if it meets the threshold.
    If words are not found or identical, WUPS is 1.0 if identical, 0.0 otherwise.
    """
    if not wordnet or not LEMMA:
        return 0.0, False # Cannot calculate if NLTK WordNet is not available

    # Simple exact match check first
    if word1.lower() == word2.lower():
        return 1.0, True

    synset1 = get_wordnet_synset(word1)
    synset2 = get_wordnet_synset(word2)

    if synset1 is None or synset2 is None:
        return 0.0, False # One or both words not in WordNet as nouns

    # Wu-Palmer similarity
    # Note: nltk.wsd.similarity_by_name uses wup_similarity internally
    # wup_similarity can return None if no common ancestor (e.g. different parts of speech, though we force NOUN)
    # or if one synset is a top-level node like 'entity.n.01' without a proper path_similarity to itself.
    # The formula is 2 * depth(lcs) / (depth(s1) + depth(s2))
    # NLTK's wup_similarity handles this. Max depth of WordNet is around 20.
    # A synset's depth is min_depth (shortest path from root).
    
    score = synset1.wup_similarity(synset2)
    if score is None: # This can happen if synsets are too dissimilar or one is a root.
        return 0.0, False
        
    return score, score >= similarity_threshold


def evaluate_model(model, data_loader, criterion_filter, criterion_vqa,
                   idx_to_answer, device,
                   lambda_filter=0.5, lambda_vqa=0.5, wups_threshold=0.9):
    model.eval()
    total_loss_filter = 0
    total_loss_vqa = 0
    total_combined_loss = 0

    all_filter_preds_idx = []
    all_filter_labels_idx = []
    
    all_vqa_preds_idx = []
    all_vqa_labels_idx = []
    
    all_vqa_preds_str = []
    all_vqa_labels_str = []
    
    all_raw_questions = []
    all_image_filenames = []
    
    wups_scores = []
    wups_at_threshold_correct = 0

    progress_bar = tqdm(data_loader, desc="Evaluating Test Set", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            img_orig = batch['image_original'].to(device)
            img_seg = batch['image_segmented'].to(device)
            img_crop = batch['image_cropped'].to(device)
            
            q_input_ids = batch['question_input_ids'].to(device)
            q_attention_mask = batch['question_attention_mask'].to(device)
            q_token_type_ids = batch['question_token_type_ids'].to(device)
            
            filter_labels = batch['filter_label'].to(device)
            vqa_labels_indices = batch['vqa_answer_label'].to(device) # Ground truth VQA indices
            
            # Store raw info for detailed results
            all_raw_questions.extend(batch['raw_question'])
            all_image_filenames.extend(batch['image_filename'])
            
            filter_logits, vqa_logits = model(
                image_original=img_orig,
                image_segmented=img_seg,
                image_cropped=img_crop,
                question_input_ids=q_input_ids,
                question_attention_mask=q_attention_mask,
                question_token_type_ids=q_token_type_ids
            )
            
            loss_filter = criterion_filter(filter_logits, filter_labels)
            loss_vqa = criterion_vqa(vqa_logits, vqa_labels_indices)
            combined_loss = (lambda_filter * loss_filter) + (lambda_vqa * loss_vqa)
            
            total_loss_filter += loss_filter.item()
            total_loss_vqa += loss_vqa.item()
            total_combined_loss += combined_loss.item()

            # Filter task predictions
            filter_preds_indices = torch.argmax(filter_logits, dim=1)
            all_filter_preds_idx.extend(filter_preds_indices.cpu().numpy())
            all_filter_labels_idx.extend(filter_labels.cpu().numpy())

            # VQA task predictions
            vqa_preds_indices_batch = torch.argmax(vqa_logits, dim=1)
            all_vqa_preds_idx.extend(vqa_preds_indices_batch.cpu().numpy())
            all_vqa_labels_idx.extend(vqa_labels_indices.cpu().numpy())

            # Convert VQA indices to strings for WUPS and qualitative results
            for pred_idx, gt_idx in zip(vqa_preds_indices_batch.cpu().numpy(), vqa_labels_indices.cpu().numpy()):
                pred_ans_str = idx_to_answer.get(pred_idx, UNK_TOKEN)
                gt_ans_str = idx_to_answer.get(gt_idx, UNK_TOKEN)
                all_vqa_preds_str.append(pred_ans_str)
                all_vqa_labels_str.append(gt_ans_str)

                if pred_ans_str != PAD_TOKEN and gt_ans_str != PAD_TOKEN and \
                   pred_ans_str != UNK_TOKEN and gt_ans_str != UNK_TOKEN: # Avoid WUPS on special tokens
                    wups_score, wups_correct = calculate_wups(pred_ans_str, gt_ans_str, wups_threshold)
                    wups_scores.append(wups_score)
                    if wups_correct:
                        wups_at_threshold_correct += 1
                else: # No valid WUPS comparison for special tokens
                    wups_scores.append(0.0) # Or handle differently, e.g. by skipping

    # --- Calculate Metrics ---
    # Filter Task
    avg_loss_filter = total_loss_filter / len(data_loader)
    filter_accuracy = accuracy_score(all_filter_labels_idx, all_filter_preds_idx)
    filter_f1_macro = f1_score(all_filter_labels_idx, all_filter_preds_idx, average='macro', zero_division=0)
    filter_f1_weighted = f1_score(all_filter_labels_idx, all_filter_preds_idx, average='weighted', zero_division=0)
    filter_report = classification_report(all_filter_labels_idx, all_filter_preds_idx, zero_division=0, output_dict=True)

    # VQA Task
    avg_loss_vqa = total_loss_vqa / len(data_loader)
    avg_combined_loss = total_combined_loss / len(data_loader)
    
    vqa_exact_match_accuracy = accuracy_score(all_vqa_labels_str, all_vqa_preds_str)
    # For F1 score, we use the indices as it's a classification over the answer vocabulary
    vqa_f1_macro = f1_score(all_vqa_labels_idx, all_vqa_preds_idx, average='macro', zero_division=0)
    vqa_f1_weighted = f1_score(all_vqa_labels_idx, all_vqa_preds_idx, average='weighted', zero_division=0)
    vqa_report_indices = classification_report(all_vqa_labels_idx, all_vqa_preds_idx, zero_division=0, output_dict=True,
                                               labels=list(idx_to_answer.keys()), target_names=list(idx_to_answer.values()))


    # WUPS Metrics
    # WUPS@0.0 (Average WUPS)
    avg_wups = np.mean(wups_scores) if wups_scores else 0.0
    # WUPS@threshold (e.g., WUPS@0.9)
    # Number of valid comparisons for WUPS@threshold (excluding those that resulted in append(0.0) due to UNK/PAD)
    num_valid_wups_comparisons = sum(1 for gt, pred in zip(all_vqa_labels_str, all_vqa_preds_str)
                                     if pred != PAD_TOKEN and gt != PAD_TOKEN and pred != UNK_TOKEN and gt != UNK_TOKEN)
    wups_accuracy_at_threshold = (wups_at_threshold_correct / num_valid_wups_comparisons) if num_valid_wups_comparisons > 0 else 0.0


    metrics = {
        "overall_combined_loss": avg_combined_loss,
        "filter_task": {
            "loss": avg_loss_filter,
            "accuracy": filter_accuracy,
            "f1_macro": filter_f1_macro,
            "f1_weighted": filter_f1_weighted,
            "classification_report": filter_report
        },
        "vqa_task": {
            "loss": avg_loss_vqa,
            "exact_match_accuracy": vqa_exact_match_accuracy,
            "f1_macro_on_indices": vqa_f1_macro,
            "f1_weighted_on_indices": vqa_f1_weighted,
            "avg_wups_score (WUPS@0.0)": avg_wups,
            f"wups_accuracy@{wups_threshold} (WUPS@{wups_threshold})": wups_accuracy_at_threshold,
            "classification_report_on_indices": vqa_report_indices
        }
    }
    
    detailed_results_df = pd.DataFrame({
        'image_filename': all_image_filenames,
        'question': all_raw_questions,
        'gt_filter_label': all_filter_labels_idx,
        'pred_filter_label': all_filter_preds_idx,
        'gt_vqa_answer_idx': all_vqa_labels_idx,
        'pred_vqa_answer_idx': all_vqa_preds_idx,
        'gt_vqa_answer_str': all_vqa_labels_str,
        'pred_vqa_answer_str': all_vqa_preds_str,
        'wups_score_vs_gt': [wups_scores[i] if i < len(wups_scores) else 0.0 for i in range(len(all_raw_questions))] # Pad if mismatch
    })

    return metrics, detailed_results_df


def main(args):
    if not download_nltk_data_if_needed():
        print("Warning: NLTK WordNet data is unavailable. WUPS scores will be 0.0.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Load answer vocabulary (answer_to_idx and idx_to_answer) ---
    try:
        with open(args.answer_vocab_path, 'r') as f:
            answer_to_idx = json.load(f)
        idx_to_answer = {int(k): v for k, v in json.load(open(args.answer_vocab_path.replace("answer_to_idx","idx_to_answer"), 'r')).items()} \
                        if os.path.exists(args.answer_vocab_path.replace("answer_to_idx","idx_to_answer")) \
                        else {v: k for k, v in answer_to_idx.items()} # Create idx_to_answer if not found
        num_vqa_classes = len(answer_to_idx)
        print(f"Loaded answer vocabulary with {num_vqa_classes} classes from {args.answer_vocab_path}")
    except Exception as e:
        print(f"Error loading answer vocabulary: {e}. Please provide a valid answer_to_idx.json.")
        return

    # --- Load Model Checkpoint and Config ---
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path {args.checkpoint_path} does not exist.")
        return
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = checkpoint.get('model_config')
    if model_config is None:
        print("Error: Model config not found in checkpoint. Please ensure it was saved during training.")
        return
    # Ensure num_vqa_answer_classes in loaded config matches the vocab
    if model_config.get('num_vqa_answer_classes') != num_vqa_classes:
        print(f"Warning: num_vqa_answer_classes in checkpoint config ({model_config.get('num_vqa_answer_classes')}) "
              f"differs from loaded vocab size ({num_vqa_classes}). Using vocab size.")
    model_config['num_vqa_answer_classes'] = num_vqa_classes


    # --- Create Test DataLoader ---
    # We need a question_type_mapping for the Dataset class
    # This should match what was used during training for consistency if filter labels are important
    question_type_map = model_config.get('question_type_mapping', {'agricultural': 1, 'other': 0, 'unknown':0}) # Example

    test_dataset = TLUFruitDataset(
        annotation_file_path=args.test_annotations,
        image_transform=DEFAULT_IMAGE_TRANSFORM, # Or load specific transform if saved in checkpoint config
        answer_to_idx=answer_to_idx,
        question_type_to_idx=question_type_map
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Test DataLoader created with {len(test_dataset)} samples.")

    # --- Initialize and Load Model ---
    model = FruitVQAModel(model_config=model_config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}.")

    # --- Define Loss Functions (mainly for consistency if any part of eval loop uses them) ---
    criterion_filter = nn.CrossEntropyLoss()
    criterion_vqa = nn.CrossEntropyLoss()

    # --- Run Evaluation ---
    print("\nStarting evaluation...")
    start_time = time.time()
    
    metrics, detailed_results_df = evaluate_model(
        model, test_loader, criterion_filter, criterion_vqa,
        idx_to_answer, device, args.lambda_filter_loss, args.lambda_vqa_loss, args.wups_threshold
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f} seconds ({eval_time/60:.2f} minutes).")

    # --- Print and Save Results ---
    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_save_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_save_path}")
    
    detailed_results_save_path = os.path.join(args.output_dir, "detailed_evaluation_results.csv")
    detailed_results_df.to_csv(detailed_results_save_path, index=False)
    print(f"Detailed results saved to: {detailed_results_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Fruit VQA Model")

    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the trained model checkpoint (.pth.tar file).")
    parser.add_argument('--test_annotations', type=str, required=True, help="Path to processed test annotations JSON file.")
    parser.add_argument('--answer_vocab_path', type=str, required=True, help="Path to the saved answer_to_idx.json file.")
    
    parser.add_argument('--output_dir', type=str, default="../evaluation_results/run01", help="Directory to save evaluation results.")
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for evaluation.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    
    # Loss weights (might not be strictly needed if only evaluating, but good for consistency if loss is reported)
    parser.add_argument('--lambda_filter_loss', type=float, default=0.5, help="Weight for the question filter loss component (for reporting).")
    parser.add_argument('--lambda_vqa_loss', type=float, default=0.5, help="Weight for the VQA answer loss component (for reporting).")
    parser.add_argument('--wups_threshold', type=float, default=0.9, help="Threshold for WUPS@ score (e.g., 0.9 for WUPS@0.9).")

    args = parser.parse_args()
    
    main(args)