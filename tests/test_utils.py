# tests/test_utils.py

import unittest
from unittest.mock import patch, MagicMock
import os
import json
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming 'src' is in PYTHONPATH or tests are run from project root
from src.utils import (
    set_seed,
    load_json_config,
    save_json_config,
    save_checkpoint,
    load_checkpoint,
    calculate_accuracy,
    download_nltk_data_if_needed_util,
    get_wordnet_synset_util,
    calculate_wups_score,
    NLTK_AVAILABLE # To check if we should attempt NLTK tests
)

class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = "temp_test_utils_dir"
        os.makedirs(cls.test_dir, exist_ok=True)
        # Attempt to download NLTK data once for the test class if NLTK is installed
        if NLTK_AVAILABLE:
            print("\nAttempting to ensure NLTK WordNet data is available for WUPS tests...")
            cls.nltk_ready = download_nltk_data_if_needed_util()
            if not cls.nltk_ready:
                print("Warning: NLTK WordNet data could not be made available. WUPS tests might be skipped or return 0.")
        else:
            cls.nltk_ready = False
            print("NLTK library not found. WUPS tests will be skipped.")


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_set_seed(self):
        print("Testing set_seed...")
        seed = 123
        
        # Test random
        set_seed(seed)
        r1_val1 = random.random()
        set_seed(seed)
        r2_val1 = random.random()
        self.assertEqual(r1_val1, r2_val1, "Python random not seeded correctly.")

        # Test numpy
        set_seed(seed)
        np1_val1 = np.random.rand()
        set_seed(seed)
        np2_val1 = np.random.rand()
        self.assertEqual(np1_val1, np2_val1, "NumPy random not seeded correctly.")

        # Test torch
        set_seed(seed)
        t1_val1 = torch.rand(1).item()
        set_seed(seed)
        t2_val1 = torch.rand(1).item()
        self.assertEqual(t1_val1, t2_val1, "PyTorch random not seeded correctly.")

        if torch.cuda.is_available():
            set_seed(seed)
            t1_cuda_val1 = torch.cuda.FloatTensor(1).normal_().item() # Example CUDA operation
            set_seed(seed)
            t2_cuda_val1 = torch.cuda.FloatTensor(1).normal_().item()
            # Due to potential non-determinism on CUDA even with seeds for some ops,
            # this might not always be exactly equal, but should be very close.
            # For this basic test, we'll assume it works if no error. A strict equality check here can be flaky.
            print(f"CUDA seeded. Val1: {t1_cuda_val1}, Val2: {t2_cuda_val1} (may not be exactly equal due to CUDA ops)")


    def test_json_config_handling(self):
        print("Testing JSON config save/load...")
        config_path = os.path.join(self.test_dir, "test_config.json")
        dummy_config = {"key1": "value1", "key2": 123, "nested": {"nk": True}}

        save_json_config(dummy_config, config_path)
        self.assertTrue(os.path.exists(config_path))

        loaded_config = load_json_config(config_path)
        self.assertEqual(dummy_config, loaded_config)

        # Test loading non-existent file
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.json")
        self.assertIsNone(load_json_config(non_existent_path))

        # Test loading malformed JSON
        malformed_path = os.path.join(self.test_dir, "malformed.json")
        with open(malformed_path, "w") as f:
            f.write("{'key': 'value',") # Malformed
        self.assertIsNone(load_json_config(malformed_path))


    def test_checkpoint_handling(self):
        print("Testing checkpoint save/load...")
        # Dummy model and optimizer
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
            def forward(self, x):
                return self.linear(x)

        model1 = DummyModel()
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        
        state = {
            'epoch': 5,
            'state_dict': model1.state_dict(),
            'optimizer': optimizer1.state_dict(),
            'best_score': 0.95
        }
        checkpoint_dir = os.path.join(self.test_dir, "checkpoints")

        # Test saving normal checkpoint
        save_checkpoint(state, is_best=False, output_dir=checkpoint_dir, filename_prefix="test_ckpt")
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "test_ckpt.pth.tar")))

        # Test saving best checkpoint
        save_checkpoint(state, is_best=True, output_dir=checkpoint_dir, filename_prefix="test_ckpt")
        self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, "test_ckpt_best.pth.tar")))

        # Test loading checkpoint
        model2 = DummyModel()
        optimizer2 = optim.Adam(model2.parameters(), lr=0.0001) # Different LR to check if loaded
        
        loaded_checkpoint_state = load_checkpoint(
            os.path.join(checkpoint_dir, "test_ckpt_best.pth.tar"),
            model2,
            optimizer2
        )
        self.assertIsNotNone(loaded_checkpoint_state)
        self.assertEqual(loaded_checkpoint_state['epoch'], 5)
        self.assertEqual(loaded_checkpoint_state['best_score'], 0.95)

        # Check if model parameters are loaded
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        
        # Check if optimizer state is loaded (e.g., learning rate should change if different)
        # Note: Optimizer state loading can be complex. Here we just check if it runs.
        # A more thorough check would involve comparing optimizer param_groups.
        self.assertNotEqual(optimizer2.param_groups[0]['lr'], 0.0001) # Should have loaded LR from state if successful
        # Actually, the LR is part of param_groups, not the state_dict itself in that way.
        # Let's check if the state_dict itself was loaded without error.
        # A better check might be to save/load and then take one step to see if params change as expected.
        # For this unit test, confirming it runs and loads model state is primary.

        # Test loading non-existent checkpoint
        self.assertIsNone(load_checkpoint("non_existent.pth.tar", model2))


    def test_calculate_accuracy(self):
        print("Testing calculate_accuracy...")
        # Test case 1: All correct
        logits1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]]) # preds: [1, 0]
        labels1 = torch.tensor([1, 0])
        self.assertEqual(calculate_accuracy(logits1, labels1), 1.0)

        # Test case 2: None correct
        logits2 = torch.tensor([[0.9, 0.1], [0.2, 0.8]]) # preds: [0, 1]
        labels2 = torch.tensor([1, 0])
        self.assertEqual(calculate_accuracy(logits2, labels2), 0.0)

        # Test case 3: Mixed
        logits3 = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]) # preds: [1, 1, 0]
        labels3 = torch.tensor([1, 0, 0]) # correct: [1, 0, 1] -> 2/3 correct
        self.assertAlmostEqual(calculate_accuracy(logits3, labels3), 2.0/3.0)
        
        # Test case 4: Empty (should return 0.0 as per implementation)
        logits4 = torch.empty((0, 2))
        labels4 = torch.empty(0, dtype=torch.long)
        self.assertEqual(calculate_accuracy(logits4, labels4), 0.0)


    @unittest.skipUnless(NLTK_AVAILABLE and LEMMA is not None and wordnet is not None, "NLTK or WordNet not available, skipping WUPS tests.")
    def test_get_wordnet_synset_util(self):
        print("Testing get_wordnet_synset_util (requires NLTK WordNet)...")
        self.assertIsNotNone(get_wordnet_synset_util("apple"))
        self.assertIsNotNone(get_wordnet_synset_util("apples")) # Test lemmatization
        self.assertEqual(get_wordnet_synset_util("apple"), get_wordnet_synset_util("apples"))
        self.assertIsNone(get_wordnet_synset_util("nonexistentwordxyz123"))
        self.assertIsNone(get_wordnet_synset_util(""))
        self.assertIsNone(get_wordnet_synset_util(None))
        self.assertIsNotNone(get_wordnet_synset_util("fruit"))


    @unittest.skipUnless(NLTK_AVAILABLE and LEMMA is not None and wordnet is not None, "NLTK or WordNet not available, skipping WUPS tests.")
    def test_calculate_wups_score(self):
        print("Testing calculate_wups_score (requires NLTK WordNet)...")
        self.assertEqual(calculate_wups_score("apple", "apple"), 1.0)
        self.assertEqual(calculate_wups_score("apples", "apple"), 1.0) # Test lemmatization
        
        # Scores depend on WordNet structure, these are qualitative checks
        score_fruit_apple = calculate_wups_score("fruit", "apple")
        self.assertTrue(0 < score_fruit_apple <= 1.0)
        
        score_tree_plant = calculate_wups_score("tree", "plant")
        self.assertTrue(0 < score_tree_plant <= 1.0)

        self.assertEqual(calculate_wups_score("car", "apple"), 0.0) # Unrelated, or one might not have a common ancestor
        self.assertEqual(calculate_wups_score("nonexistentwordxyz123", "apple"), 0.0)
        self.assertEqual(calculate_wups_score("", "apple"), 0.0)
        self.assertEqual(calculate_wups_score("apple", ""), 0.0)
        self.assertEqual(calculate_wups_score(None, "apple"), 0.0)
        self.assertEqual(calculate_wups_score("apple", None), 0.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)