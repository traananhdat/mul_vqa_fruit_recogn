# src/utils.py

import os
import json
import random
import numpy as np
import torch

# For WUPS score
import nltk
try:
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not found. WUPS calculation will not be available. Please install it: pip install nltk")
    NLTK_AVAILABLE = False
    wordnet = None
    WordNetLemmatizer = None

# Initialize lemmatizer globally if NLTK is available
LEMMA = None
if NLTK_AVAILABLE:
    LEMMA = WordNetLemmatizer()

# --- Reproducibility ---
def set_seed(seed_value=42):
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # The two lines below are often used for deterministic behavior with CUDA,
        # but can impact performance. Use if exact reproducibility is critical.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed_value}")

# --- Configuration Handling ---
def load_json_config(config_path):
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from config file: {config_path}")
        return None

def save_json_config(config_dict, config_path):
    """Saves a dictionary to a JSON configuration file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to: {config_path}")
    except Exception as e:
        print(f"ERROR saving configuration to {config_path}: {e}")

# --- Checkpoint Management ---
def save_checkpoint(state, is_best, output_dir, filename_prefix="checkpoint"):
    """
    Saves model and training parameters at checkpoint.
    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): True if it's the best model seen so far.
        output_dir (str): Directory to save the checkpoint.
        filename_prefix (str): Prefix for the checkpoint file names.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename_prefix}.pth.tar")
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    if is_best:
        best_filepath = os.path.join(output_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)
        print(f"Saved new best model to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Loads model parameters (and optionally optimizer/scheduler state) from a checkpoint.
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance.
        optimizer (torch.optim.Optimizer, optional): Optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler instance.
        device (str or torch.device): Device to load the model to.
    Returns:
        dict: The loaded checkpoint dictionary (can be useful for epoch, best_score, etc.).
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Model not loaded.")
        return None

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    print("Model weights loaded successfully.")

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Optimizer may start fresh.")
            
    if scheduler and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Scheduler state loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}. Scheduler may start fresh.")
            
    return checkpoint


# --- Metric Calculation ---
def calculate_accuracy(logits, labels):
    """Calculates accuracy for classification tasks."""
    if logits.is_cuda: # Ensure labels are on the same device for comparison
        labels = labels.to(logits.device)
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


# --- Wu-Palmer Similarity (WUPS) Calculation ---
_nltk_data_downloaded = False

def download_nltk_data_if_needed_util():
    """Downloads NLTK WordNet if not already present. Internal to utils."""
    global _nltk_data_downloaded
    if not NLTK_AVAILABLE or _nltk_data_downloaded:
        return NLTK_AVAILABLE

    try:
        wordnet.ensure_loaded() # Quick check
        _nltk_data_downloaded = True
        print("WordNet is already available (utils).")
        return True
    except LookupError:
        print("WordNet data not found (utils). Attempting to download...")
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True) # Open Multilingual WordNet
            wordnet.ensure_loaded() # Re-check after download
            _nltk_data_downloaded = True
            print("WordNet downloaded successfully (utils).")
            # Re-initialize LEMMA in case NLTK was imported but data was missing before
            global LEMMA
            if WordNetLemmatizer: LEMMA = WordNetLemmatizer()
            return True
        except Exception as e:
            print(f"Failed to download WordNet (utils): {e}")
            print("Please download WordNet manually: import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')")
            return False

def get_wordnet_synset_util(word):
    """Get the first noun synset for a word, lemmatizing first. Internal to utils."""
    if not NLTK_AVAILABLE or not LEMMA or not wordnet:
        return None
    if not word: # Handle empty string
        return None
    # Ensure NLTK data is ready before attempting to use lemmatizer or synsets
    if not _nltk_data_downloaded:
        download_nltk_data_if_needed_util()
        if not _nltk_data_downloaded: # Still not available
            return None

    try:
        lemma = LEMMA.lemmatize(word.lower().strip(), pos='n') # Lemmatize as noun
        synsets = wordnet.synsets(lemma, pos=wordnet.NOUN)
        return synsets[0] if synsets else None
    except Exception as e: # Catch any NLTK internal errors if data is partially corrupt
        print(f"Error in NLTK processing for word '{word}': {e}")
        return None


def calculate_wups_score(word1, word2):
    """
    Calculates Wu-Palmer Similarity (WUPS) between two words.
    Returns the WUPS score.
    If words are not found, not comparable, or identical, WUPS is 1.0 if identical, 0.0 otherwise.
    """
    if not NLTK_AVAILABLE:
        # print("NLTK WordNet not available, returning WUPS = 0.0")
        return 0.0

    s_word1 = str(word1).lower().strip()
    s_word2 = str(word2).lower().strip()

    if not s_word1 or not s_word2: # Handle empty strings after stripping
        return 0.0
    if s_word1 == s_word2:
        return 1.0

    synset1 = get_wordnet_synset_util(s_word1)
    synset2 = get_wordnet_synset_util(s_word2)

    if synset1 is None or synset2 is None:
        return 0.0 # One or both words not in WordNet as nouns or error during synset retrieval

    score = synset1.wup_similarity(synset2)
    
    # wup_similarity can return None if no common ancestor or other issues.
    return score if score is not None else 0.0


if __name__ == '__main__':
    # Example usage of utility functions (optional, for testing utils.py itself)
    print("--- Testing utils.py ---")

    # Seed
    set_seed(123)
    print(f"Random float after seed: {random.random()}")

    # Config
    dummy_config = {"learning_rate": 0.001, "model_type": "test_model"}
    config_path = "dummy_config_test.json"
    save_json_config(dummy_config, config_path)
    loaded_cfg = load_json_config(config_path)
    assert loaded_cfg == dummy_config
    print(f"Config load/save test passed with: {loaded_cfg}")
    if os.path.exists(config_path):
        os.remove(config_path)

    # Checkpoint (mocking)
    mock_state = {'epoch': 1, 'model_state': {'param': torch.tensor([1.0])}}
    output_dir_test = "dummy_checkpoints_test"
    save_checkpoint(mock_state, True, output_dir_test, "test_model_ckpt")
    save_checkpoint(mock_state, False, output_dir_test, "test_model_ckpt_epoch2")
    if os.path.exists(output_dir_test):
        for f in os.listdir(output_dir_test):
            os.remove(os.path.join(output_dir_test, f))
        os.rmdir(output_dir_test)
    print("Checkpoint saving functions called (check console for messages).")


    # Accuracy
    logits_test = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    labels_test = torch.tensor([1, 0, 0]) # Correct preds: 1, 0. Incorrect: 1 (pred 1, actual 0)
    acc = calculate_accuracy(logits_test, labels_test)
    print(f"Accuracy test: {acc} (Expected around 0.666...)") # Should be 2/3 = 0.666...

    # WUPS
    if NLTK_AVAILABLE:
        download_nltk_data_if_needed_util() # Ensure data is there for tests
        print(f"WUPS('apple', 'apple'): {calculate_wups_score('apple', 'apple')}")
        print(f"WUPS('apple', 'banana'): {calculate_wups_score('apple', 'banana')}")
        print(f"WUPS('fruit', 'apple'): {calculate_wups_score('fruit', 'apple')}")
        print(f"WUPS('car', 'apple'): {calculate_wups_score('car', 'apple')}")
        print(f"WUPS('tree', 'plant'): {calculate_wups_score('tree', 'plant')}")
        print(f"WUPS('xyz', 'apple'): {calculate_wups_score('xyz123', 'apple')}") # Non-word
        print(f"WUPS('', 'apple'): {calculate_wups_score('', 'apple')}") # Empty string
    else:
        print("Skipping WUPS examples as NLTK is not fully available.")

    print("--- utils.py testing finished ---")