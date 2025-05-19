# tests/test_data_loader.py

import unittest
import os
import json
import shutil
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

# Assuming 'src' is in PYTHONPATH or tests are run from project root
# If running tests directly from 'tests' dir and 'src' is a sibling:
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import (
    TLUFruitDataset,
    build_answer_vocab,
    create_data_loaders,
    DEFAULT_IMAGE_TRANSFORM,
    UNK_TOKEN,
    PAD_TOKEN
)

# Constants for testing
MAX_TOKEN_LENGTH_TEST = 32 # Should match preprocessing for dummy data consistency
IMG_SIZE_TEST = (64, 64) # Smaller images for faster tests
DUMMY_IMAGE_TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(IMG_SIZE_TEST), # Resize for consistency, though preproc should do it
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test data
        cls.test_dir = "temp_test_data_loader_dir"
        cls.raw_images_dir = os.path.join(cls.test_dir, "raw_images") # To store dummy original images
        cls.processed_images_dir_orig = os.path.join(cls.test_dir, "processed_images", "original")
        cls.processed_images_dir_seg = os.path.join(cls.test_dir, "processed_images", "segmented")
        cls.processed_images_dir_crop = os.path.join(cls.test_dir, "processed_images", "cropped")
        cls.annotations_dir = os.path.join(cls.test_dir, "annotations")

        os.makedirs(cls.raw_images_dir, exist_ok=True)
        os.makedirs(cls.processed_images_dir_orig, exist_ok=True)
        os.makedirs(cls.processed_images_dir_seg, exist_ok=True)
        os.makedirs(cls.processed_images_dir_crop, exist_ok=True)
        os.makedirs(cls.annotations_dir, exist_ok=True)

        # Create dummy images
        cls.dummy_image_names = ["img1.png", "img2.png", "img3.png"]
        for img_name in cls.dummy_image_names:
            try:
                # Raw image (source for "processing")
                raw_img_path = os.path.join(cls.raw_images_dir, img_name)
                Image.new('RGB', (100, 100), color='blue').save(raw_img_path)

                # "Processed" images (just copies for testing paths)
                Image.open(raw_img_path).resize(IMG_SIZE_TEST).save(os.path.join(cls.processed_images_dir_orig, img_name))
                Image.open(raw_img_path).resize(IMG_SIZE_TEST).save(os.path.join(cls.processed_images_dir_seg, f"{os.path.splitext(img_name)[0]}_seg{os.path.splitext(img_name)[1]}"))
                Image.open(raw_img_path).resize(IMG_SIZE_TEST).save(os.path.join(cls.processed_images_dir_crop, f"{os.path.splitext(img_name)[0]}_crop{os.path.splitext(img_name)[1]}"))
            except Exception as e:
                print(f"Warning: Could not create dummy image {img_name} for tests: {e}")


        # Create dummy annotation data (mimicking output of 02_preprocessing script)
        cls.dummy_train_data = [
            {
                "image_filename": "img1.png",
                "original_processed_path": os.path.join(cls.processed_images_dir_orig, "img1.png"),
                "segmented_processed_path": os.path.join(cls.processed_images_dir_seg, "img1_seg.png"),
                "cropped_processed_path": os.path.join(cls.processed_images_dir_crop, "img1_crop.png"),
                "question": "what fruit is this",
                "answer": "apple", # Common answer
                "question_type": "agricultural",
                "question_input_ids": [101, 2054, 5 frutas, 2003, 2023, 102] + [0]*(MAX_TOKEN_LENGTH_TEST-6),
                "question_attention_mask": [1]*6 + [0]*(MAX_TOKEN_LENGTH_TEST-6),
                "question_token_type_ids": [0]*MAX_TOKEN_LENGTH_TEST,
            },
            {
                "image_filename": "img2.png",
                "original_processed_path": os.path.join(cls.processed_images_dir_orig, "img2.png"),
                "segmented_processed_path": os.path.join(cls.processed_images_dir_seg, "img2_seg.png"),
                "cropped_processed_path": os.path.join(cls.processed_images_dir_crop, "img2_crop.png"),
                "question": "how many are there",
                "answer": "banana", # Another common answer
                "question_type": "other",
                "question_input_ids": [101, 2129, 2116, 2024, 2073, 102] + [0]*(MAX_TOKEN_LENGTH_TEST-6),
                "question_attention_mask": [1]*6 + [0]*(MAX_TOKEN_LENGTH_TEST-6),
                "question_token_type_ids": [0]*MAX_TOKEN_LENGTH_TEST,
            },
            {
                "image_filename": "img1.png", # Same image, different QA
                "original_processed_path": os.path.join(cls.processed_images_dir_orig, "img1.png"),
                "segmented_processed_path": os.path.join(cls.processed_images_dir_seg, "img1_seg.png"),
                "cropped_processed_path": os.path.join(cls.processed_images_dir_crop, "img1_crop.png"),
                "question": "is it red",
                "answer": "apple", # Common again
                "question_type": "agricultural",
                "question_input_ids": [101, 2003, 2009, 2735, 102, 0] + [0]*(MAX_TOKEN_LENGTH_TEST-6),
                "question_attention_mask": [1]*5 + [0]*(MAX_TOKEN_LENGTH_TEST-5),
                "question_token_type_ids": [0]*MAX_TOKEN_LENGTH_TEST,
            },
             { # For UNK testing
                "image_filename": "img3.png",
                "original_processed_path": os.path.join(cls.processed_images_dir_orig, "img3.png"),
                "segmented_processed_path": os.path.join(cls.processed_images_dir_seg, "img3_seg.png"),
                "cropped_processed_path": os.path.join(cls.processed_images_dir_crop, "img3_crop.png"),
                "question": "what is this rare thing",
                "answer": "grapefruit", # Assume this will be UNK
                "question_type": "agricultural",
                "question_input_ids": [101, 2054, 2003, 2023, 6244, 2518, 102] + [0]*(MAX_TOKEN_LENGTH_TEST-7),
                "question_attention_mask": [1]*7 + [0]*(MAX_TOKEN_LENGTH_TEST-7),
                "question_token_type_ids": [0]*MAX_TOKEN_LENGTH_TEST,
            }
        ]
        cls.train_ann_path = os.path.join(cls.annotations_dir, "dummy_train_ann.json")
        with open(cls.train_ann_path, 'w') as f:
            json.dump(cls.dummy_train_data, f)

        # Create dummy val/test data (can be same as train for simplicity of test setup)
        cls.val_ann_path = os.path.join(cls.annotations_dir, "dummy_val_ann.json")
        with open(cls.val_ann_path, 'w') as f:
            json.dump(cls.dummy_train_data[:2], f) # Smaller val set

        cls.test_ann_path = os.path.join(cls.annotations_dir, "dummy_test_ann.json")
        with open(cls.test_ann_path, 'w') as f:
            json.dump(cls.dummy_train_data[2:], f) # Smaller test set
            
        cls.question_type_map_test = {'agricultural': 1, 'other': 0, 'unknown': 0}


    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory after all tests are done
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_build_answer_vocab(self):
        print("Testing build_answer_vocab...")
        train_df = pd.DataFrame(self.dummy_train_data)
        answer_to_idx, idx_to_answer = build_answer_vocab(train_df, top_k=1) # top_k=1 means 'apple'

        self.assertEqual(len(answer_to_idx), 1 + 2) # apple + UNK + PAD
        self.assertEqual(len(idx_to_answer), 1 + 2)
        self.assertIn("apple", answer_to_idx)
        self.assertIn(UNK_TOKEN, answer_to_idx)
        self.assertIn(PAD_TOKEN, answer_to_idx)
        self.assertEqual(answer_to_idx[PAD_TOKEN], 0)
        self.assertEqual(answer_to_idx[UNK_TOKEN], 1)
        self.assertEqual(answer_to_idx["apple"], 2)

    def test_tlu_fruit_dataset_init_len(self):
        print("Testing TLUFruitDataset initialization and length...")
        train_df = pd.DataFrame(self.dummy_train_data)
        answer_to_idx, _ = build_answer_vocab(train_df, top_k=2)

        dataset = TLUFruitDataset(
            annotation_file_path=self.train_ann_path,
            image_transform=DUMMY_IMAGE_TRANSFORM_TEST,
            answer_to_idx=answer_to_idx,
            question_type_to_idx=self.question_type_map_test
        )
        self.assertEqual(len(dataset), len(self.dummy_train_data))

    def test_tlu_fruit_dataset_getitem(self):
        print("Testing TLUFruitDataset __getitem__...")
        train_df = pd.DataFrame(self.dummy_train_data)
        answer_to_idx, _ = build_answer_vocab(train_df, top_k=1) # 'apple' is top 1, 'banana' will be UNK for this vocab

        dataset = TLUFruitDataset(
            annotation_file_path=self.train_ann_path,
            image_transform=DUMMY_IMAGE_TRANSFORM_TEST,
            answer_to_idx=answer_to_idx,
            question_type_to_idx=self.question_type_map_test
        )
        
        # Test first sample ("apple", "agricultural")
        sample0 = dataset[0]
        self.assertIsInstance(sample0, dict)
        self.assertEqual(sample0['image_original'].shape, (3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))
        self.assertEqual(sample0['image_segmented'].shape, (3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))
        self.assertEqual(sample0['image_cropped'].shape, (3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))
        self.assertEqual(sample0['question_input_ids'].shape, (MAX_TOKEN_LENGTH_TEST,))
        self.assertEqual(sample0['vqa_answer_label'].item(), answer_to_idx["apple"])
        self.assertEqual(sample0['filter_label'].item(), self.question_type_map_test["agricultural"])

        # Test second sample ("banana" -> UNK, "other")
        sample1 = dataset[1]
        self.assertEqual(sample1['vqa_answer_label'].item(), answer_to_idx[UNK_TOKEN])
        self.assertEqual(sample1['filter_label'].item(), self.question_type_map_test["other"])
        
        # Test fourth sample ("grapefruit" -> UNK)
        sample3 = dataset[3]
        self.assertEqual(sample3['vqa_answer_label'].item(), answer_to_idx[UNK_TOKEN])


    def test_tlu_fruit_dataset_missing_image_fallback(self):
        print("Testing TLUFruitDataset missing aux image fallback...")
        # Create a temporary annotation with a non-existent segmented image path
        faulty_data = [{
            "image_filename": "img1.png",
            "original_processed_path": os.path.join(self.processed_images_dir_orig, "img1.png"),
            "segmented_processed_path": os.path.join(self.processed_images_dir_seg, "non_existent_seg.png"), # This file won't exist
            "cropped_processed_path": os.path.join(self.processed_images_dir_crop, "img1_crop.png"),
            "question": "test q", "answer": "apple", "question_type": "agricultural",
            "question_input_ids": [101,0,102] + [0]*(MAX_TOKEN_LENGTH_TEST-3), "question_attention_mask": [1]*3 + [0]*(MAX_TOKEN_LENGTH_TEST-3), "question_token_type_ids": [0]*MAX_TOKEN_LENGTH_TEST,
        }]
        faulty_ann_path = os.path.join(self.annotations_dir, "faulty_ann.json")
        with open(faulty_ann_path, 'w') as f: json.dump(faulty_data, f)

        train_df = pd.DataFrame(self.dummy_train_data) # For vocab
        answer_to_idx, _ = build_answer_vocab(train_df, top_k=1)
        
        dataset = TLUFruitDataset(
            annotation_file_path=faulty_ann_path,
            image_transform=DUMMY_IMAGE_TRANSFORM_TEST,
            answer_to_idx=answer_to_idx,
            question_type_to_idx=self.question_type_map_test
        )
        # This should trigger the warning and fallback in __getitem__
        # The test here is that it *doesn't crash* and image_segmented tensor is still returned
        # (it would be a copy of image_original)
        try:
            sample = dataset[0]
            self.assertIsNotNone(sample['image_segmented'])
            self.assertEqual(sample['image_segmented'].shape, (3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))
            # For a more rigorous test, you'd compare if sample['image_segmented'] equals sample['image_original']
            # but that involves comparing float tensors which can be tricky. Shape check is a good start.
        except Exception as e:
            self.fail(f"Dataset __getitem__ crashed with missing aux image: {e}")


    def test_create_data_loaders(self):
        print("Testing create_data_loaders...")
        data_loader_config = {
            'train_annotations_path': self.train_ann_path,
            'val_annotations_path': self.val_ann_path,
            'test_annotations_path': self.test_ann_path,
            'batch_size': 2,
            'num_workers': 0, # Use 0 for easier debugging in tests
            'top_k_answers': 1, # 'apple' + UNK + PAD
            'question_type_mapping': self.question_type_map_test
        }
        
        train_loader, val_loader, test_loader, num_classes, answer_vocab = create_data_loaders(
            data_loader_config,
            image_transform=DUMMY_IMAGE_TRANSFORM_TEST
        )

        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        self.assertEqual(num_classes, 1 + 2) # apple + UNK + PAD
        self.assertIn("apple", answer_vocab)

        # Check batch from train_loader
        train_batch = next(iter(train_loader))
        self.assertEqual(train_batch['image_original'].shape, (2, 3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))
        self.assertEqual(train_batch['question_input_ids'].shape, (2, MAX_TOKEN_LENGTH_TEST))
        self.assertEqual(train_batch['vqa_answer_label'].shape, (2,))
        self.assertEqual(train_batch['filter_label'].shape, (2,))

        # Check batch from val_loader
        val_batch = next(iter(val_loader))
        self.assertEqual(len(val_loader.dataset), 2) # dummy_train_data[:2]
        self.assertEqual(val_batch['image_original'].shape, (2, 3, IMG_SIZE_TEST[0], IMG_SIZE_TEST[1]))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)