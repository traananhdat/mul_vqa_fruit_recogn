# src/predict.py

import argparse
import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizerFast

# Assuming model_architecture.py is in the same src directory
from .model_architecture import FruitVQAModel # DEFAULT_CONFIG is also available if needed
from .data_loader import DEFAULT_IMAGE_TRANSFORM # Using the same transform as in data_loader

def load_model_from_checkpoint(checkpoint_path, device):
    """Loads the model, its config, and answer vocabulary from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get('model_config')
    if model_config is None:
        raise ValueError("Model config not found in checkpoint.")

    answer_to_idx = checkpoint.get('answer_to_idx')
    if answer_to_idx is None:
        raise ValueError("answer_to_idx mapping not found in checkpoint. Please use a checkpoint from train.py that saves it, or provide a separate vocab file.")
    
    idx_to_answer = {int(k): v for k, v in answer_to_idx.items()} # Ensure keys are int if loaded from json string keys

    # Update num_vqa_answer_classes in model_config based on loaded vocab
    num_vqa_classes = len(answer_to_idx)
    if model_config.get('num_vqa_answer_classes') != num_vqa_classes:
        print(f"Warning: num_vqa_answer_classes in checkpoint config ({model_config.get('num_vqa_answer_classes')}) "
              f"differs from loaded vocab size ({num_vqa_classes}). Using vocab size from loaded answer_to_idx.")
    model_config['num_vqa_answer_classes'] = num_vqa_classes
    
    model = FruitVQAModel(model_config=model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")
    return model, model_config, idx_to_answer, answer_to_idx

def preprocess_inputs(original_image_path, question_text, tokenizer, image_transform, device,
                      max_token_length, segmented_image_path=None, cropped_image_path=None):
    """Preprocesses image and text inputs for the model."""

    # --- Image Preprocessing ---
    try:
        img_orig = Image.open(original_image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Original image not found: {original_image_path}")

    # Handle segmented image
    if segmented_image_path and os.path.exists(segmented_image_path):
        img_seg = Image.open(segmented_image_path).convert("RGB")
        print(f"Using provided segmented image: {segmented_image_path}")
    else:
        if segmented_image_path: # Path was given but not found
             print(f"Warning: Segmented image not found at '{segmented_image_path}'. Using original image as fallback for segmented stream.")
        else: # Path was not given
             print("Warning: Segmented image path not provided. Using original image as fallback for segmented stream.")
        img_seg = img_orig.copy() # Use original as fallback

    # Handle cropped image
    if cropped_image_path and os.path.exists(cropped_image_path):
        img_crop = Image.open(cropped_image_path).convert("RGB")
        print(f"Using provided cropped image: {cropped_image_path}")
    else:
        if cropped_image_path: # Path was given but not found
            print(f"Warning: Cropped image not found at '{cropped_image_path}'. Using original image as fallback for cropped stream.")
        else: # Path was not given
            print("Warning: Cropped image path not provided. Using original image as fallback for cropped stream.")
        img_crop = img_orig.copy() # Use original as fallback

    # Apply transformations
    img_orig_tensor = image_transform(img_orig).unsqueeze(0).to(device) # Add batch dimension
    img_seg_tensor = image_transform(img_seg).unsqueeze(0).to(device)
    img_crop_tensor = image_transform(img_crop).unsqueeze(0).to(device)

    # --- Text Preprocessing ---
    tokenized_question = tokenizer(
        question_text,
        padding='max_length',
        truncation=True,
        max_length=max_token_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )
    q_input_ids = tokenized_question['input_ids'].to(device)
    q_attention_mask = tokenized_question['attention_mask'].to(device)
    q_token_type_ids = tokenized_question['token_type_ids'].to(device)

    return (img_orig_tensor, img_seg_tensor, img_crop_tensor,
            q_input_ids, q_attention_mask, q_token_type_ids)


def predict(model, processed_inputs, idx_to_answer, top_k=5):
    """Makes a prediction using the model and processed inputs."""
    (img_orig_tensor, img_seg_tensor, img_crop_tensor,
     q_input_ids, q_attention_mask, q_token_type_ids) = processed_inputs

    with torch.no_grad():
        filter_logits, vqa_logits = model(
            image_original=img_orig_tensor,
            image_segmented=img_seg_tensor,
            image_cropped=img_crop_tensor,
            question_input_ids=q_input_ids,
            question_attention_mask=q_attention_mask,
            question_token_type_ids=q_token_type_ids
        )

    # --- Process Filter Output ---
    filter_probs = F.softmax(filter_logits, dim=-1)
    filter_confidence, filter_pred_idx = torch.max(filter_probs, dim=-1)
    # Assuming filter_idx_to_class = {0: "Non-Agricultural", 1: "Agricultural"}
    # This mapping should ideally come from model_config or training args
    filter_idx_to_class = model.config.get('question_filter_idx_to_class', {0: "Other/Non-Relevant", 1: "Agricultural/Fruit-Related"})
    predicted_filter_class = filter_idx_to_class.get(filter_pred_idx.item(), "Unknown Filter Class")
    filter_result = {
        "class": predicted_filter_class,
        "confidence": filter_confidence.item()
    }

    # --- Process VQA Output ---
    vqa_probs = F.softmax(vqa_logits, dim=-1)
    top_k_vqa_probs, top_k_vqa_indices = torch.topk(vqa_probs, k=top_k, dim=-1)
    
    vqa_results = []
    for i in range(top_k):
        pred_idx = top_k_vqa_indices[0, i].item() # Batch size is 1
        pred_answer_str = idx_to_answer.get(pred_idx, "<UNK_ANSWER>")
        pred_confidence = top_k_vqa_probs[0, i].item()
        vqa_results.append({
            "answer": pred_answer_str,
            "confidence": pred_confidence
        })
        
    return filter_result, vqa_results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Vocab ---
    try:
        model, model_config, idx_to_answer, _ = load_model_from_checkpoint(args.checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model or vocabulary: {e}")
        return

    # --- Initialize Tokenizer and Image Transform ---
    bert_model_name = model_config.get('bert_model_name', 'bert-base-uncased') # Get from loaded config
    max_token_len = model_config.get('max_token_length', 128) # Get from loaded config or use default
    image_size = (model_config.get('vit_img_size', 224), model_config.get('vit_img_size', 224))

    try:
        tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
    except Exception as e:
        print(f"Error loading tokenizer '{bert_model_name}': {e}")
        return
        
    # Use the same transform as training, potentially simplified if not all details are in model_config
    # DEFAULT_IMAGE_TRANSFORM already defined with resize, ToTensor, Normalize
    # We need to ensure resize matches the model's expected input size
    current_image_transform = transforms.Compose([
        transforms.Resize(image_size), # Use image_size from model_config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # --- Preprocess Inputs ---
    print("\nPreprocessing inputs...")
    try:
        processed_inputs = preprocess_inputs(
            args.original_image_path,
            args.question,
            tokenizer,
            current_image_transform,
            device,
            max_token_len,
            args.segmented_image_path,
            args.cropped_image_path
        )
    except Exception as e:
        print(f"Error during input preprocessing: {e}")
        return

    # --- Make Prediction ---
    print("Making prediction...")
    start_time = time.time()
    try:
        filter_result, vqa_results = predict(model, processed_inputs, idx_to_answer, top_k=args.top_k_answers)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    prediction_time = time.time() - start_time
    print(f"Prediction completed in {prediction_time:.4f} seconds.")

    # --- Display Results ---
    print("\n--- Prediction Results ---")
    print(f"Input Question: \"{args.question}\"")
    
    print("\nQuestion Filter Task:")
    print(f"  Predicted Category: {filter_result['class']}")
    print(f"  Confidence: {filter_result['confidence']:.4f}")
    
    print("\nVQA Task (Top {} Answers):".format(args.top_k_answers))
    for i, res in enumerate(vqa_results):
        print(f"  {i+1}. Answer: \"{res['answer']}\" (Confidence: {res['confidence']:.4f})")

    # --- Optionally save results to a JSON file ---
    if args.output_file:
        results_to_save = {
            "input_original_image": args.original_image_path,
            "input_segmented_image": args.segmented_image_path if args.segmented_image_path else "N/A (used original)",
            "input_cropped_image": args.cropped_image_path if args.cropped_image_path else "N/A (used original)",
            "input_question": args.question,
            "predicted_filter_category": filter_result,
            "predicted_vqa_answers": vqa_results,
            "model_checkpoint": args.checkpoint_path,
            "prediction_time_seconds": prediction_time
        }
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using a trained Fruit VQA Model.")

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the trained model checkpoint (.pth.tar file).")
    # The answer_vocab_path is now effectively handled by being part of the checkpoint.
    # If not, it would need to be a separate argument.
    # parser.add_argument('--answer_vocab_path', type=str, required=True,
    #                     help="Path to the saved answer_to_idx.json file.")

    parser.add_argument('--original_image_path', type=str, required=True,
                        help="Path to the original input image file.")
    parser.add_argument('--question', type=str, required=True,
                        help="Question text to ask about the image.")
    
    parser.add_argument('--segmented_image_path', type=str, default=None,
                        help="(Optional) Path to the pre-segmented version of the input image.")
    parser.add_argument('--cropped_image_path', type=str, default=None,
                        help="(Optional) Path to the pre-cropped version of the input image (focusing on the fruit/object).")

    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for prediction (cuda or cpu).")
    parser.add_argument('--top_k_answers', type=int, default=3,
                        help="Number of top VQA answers to display.")
    parser.add_argument('--output_file', type=str, default=None,
                        help="(Optional) Path to save the prediction results as a JSON file.")
    
    # Max token length and BERT model name can be inferred from model_config in checkpoint.
    # Adding them as args would be for overriding or if config is missing these details.
    # parser.add_argument('--max_token_length', type=int, default=128, help="Max sequence length for BERT tokenizer.")
    # parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased', help="Name of the BERT model for tokenizer.")


    args = parser.parse_args()
    main(args)