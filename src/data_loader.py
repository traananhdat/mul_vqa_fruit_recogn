# src/data_loader.py

import os
import json
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Default ImageNet normalization, as ViT/CNN backbones are often pretrained on it
DEFAULT_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Special tokens for answer vocabulary
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class TLUFruitDataset(Dataset):
    def __init__(self, annotation_file_path, image_transform,
                 answer_to_idx, question_type_to_idx,
                 is_train=False): # is_train might be used for specific augmentations later
        """
        Args:
            annotation_file_path (str): Path to the processed JSON annotation file.
            image_transform (callable): Torchvision transforms to be applied to each