# Transformer-guided Multimodal VQA for Fruit Recognition

This project implements the model described in the paper "Transformer guided Multimodal VQA model for Fruit recognitions" by Dat Tran and Hoai Nam Vu. The model is designed for Visual Question Answering (VQA) tasks, with a specific focus on agricultural applications like fruit recognition and the detection of counterfeit agricultural products.

## Overview

The core of this project is a novel multi-stream, multi-modal Transformer architecture. Key architectural innovations include:
1.  **Concurrent Visual Processing:** Utilizes multiple image representations (original, segmented, and cropped) processed via a CNN-ViT like pipeline.
2.  **PCA Enhancement:** Visual features are enhanced by PCA-based feature transformation.
3.  **BERT for Text:** Employs BERT-based encoding for textual queries.
4.  **Dedicated Fusion Module:** A fusion module integrates the processed visual inputs (post-PCA) with BERT-encoded textual data.
5.  **Multi-task Learning:** The architecture supports a multi-task framework, including domain-specific question filtering (e.g., identifying if a question is agriculture-related) before addressing the primary VQA task.

This codebase provides tools for data preprocessing, model definition, training, evaluation, and making predictions.

## Setup and Installation

### Prerequisites
* Python 3.8+
* pip

### Steps
1.  **Clone the repository (if applicable):**
    ```bash
    git clone [your-repository-url]
    cd fruit-vqa-project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data (for WUPS score in evaluation):**
    Run the following in a Python interpreter if you haven't already:
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual WordNet
    ```
    The `src/utils.py` and `src/evaluate.py` scripts also attempt to trigger this download if data is missing.

## Dataset Preparation

This project uses two main datasets:
* **TLU-Fruit:** A custom dataset for agricultural VQA introduced in the paper. You will need to have access to this dataset.
* **COCO-QA:** A standard VQA benchmark. You can find instructions for downloading COCO images and COCO-QA annotations from their respective official sources.

1.  **Place Raw Data:**
    * Organize your raw TLU-Fruit images and annotation file (e.g., `annotations_tlu_fruit.json`) under `data/raw/TLU-Fruit/images/` and `data/raw/TLU-Fruit/` respectively.
    * Similarly, for COCO-QA, place its annotations and relevant COCO images in a structured way that your preprocessing script can access (e.g., under `data/raw/COCO-QA/`).

2.  **Run Preprocessing:**
    * The `notebooks/02_preprocessing_TLU-Fruit.ipynb` notebook (or a script version of it) should be run to process the raw data. This includes:
        * Tokenizing questions and answers.
        * Preparing image paths (original, and placeholders/actual paths for segmented and cropped versions).
        * Splitting data into train, validation, and test sets.
    * The preprocessed annotation files (e.g., `train_tlu_fruit_processed.json`) will be saved to `data/processed/TLU-Fruit/`.
    * Processed images (e.g., resized versions) might be saved to `data/interim/TLU-Fruit/`.
    * **Note on Segmented/Cropped Images:** The paper mentions using Segment Anything Model (SAM) for segmentation and also using cropped images. The provided preprocessing notebook creates placeholders for these image paths. You will need to:
        * Generate these segmented and cropped images using appropriate tools/scripts.
        * Ensure the paths in your preprocessed annotation files correctly point to these generated images, or modify the `data_loader.py` to handle their generation/loading if paths are just indicative. The current `data_loader.py` expects these files to exist if paths are provided, or uses fallbacks for prediction/evaluation if paths are invalid/missing.

## Configuration

The project uses YAML configuration files located in `src/configs/`.
* **`base_config.yaml`**: Contains default settings for data, model architecture, training, and evaluation.
* **`tlu_fruit_config.yaml`**: Experiment-specific configurations for the TLU-Fruit dataset, overriding base settings.
* **`coco_qa_config.yaml`**: Experiment-specific configurations for the COCO-QA dataset.

You can create new experiment configuration files or modify existing ones. The training/evaluation scripts typically load a base config and then merge it with an experiment-specific config. Command-line arguments can further override these settings.

## How to Run

Make sure you are in the project's root directory and your virtual environment is activated.

### 1. Training

Use the `src/train.py` script. You'll need to provide paths to your preprocessed annotation files and specify an output directory. Configuration parameters can be loaded from a YAML file and overridden by CLI arguments.

**Example for TLU-Fruit:**
```bash
python -m src.train \
    --train_annotations data/processed/TLU-Fruit/train_tlu_fruit_processed.json \
    --val_annotations data/processed/TLU-Fruit/val_tlu_fruit_processed.json \
    --model_config_file src/configs/tlu_fruit_config.yaml \
    --output_dir experiments/tlu_fruit_experiment_01 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 5e-5
```

### 2. Evaluation

```bash
python -m src.evaluate \
    --checkpoint_path experiments/tlu_fruit_experiment_01/fruit_vqa_epoch_XX_best.pth.tar \
    --test_annotations data/processed/TLU-Fruit/test_tlu_fruit_processed.json \
    --answer_vocab_path experiments/tlu_fruit_experiment_01/answer_to_idx.json \
    --output_dir experiments/tlu_fruit_experiment_01/evaluation_results \
    --wups_threshold 0.9
```

### 3. Prediction

```bash
python -m src.predict \
    --checkpoint_path experiments/tlu_fruit_experiment_01/fruit_vqa_epoch_XX_best.pth.tar \
    --original_image_path path/to/your/image.jpg \
    --question "What type of fruit is this?" \
    --top_k_answers 3 \
    --output_file predictions/single_prediction_result.json
    # --segmented_image_path path/to/your/segmented_image.jpg \ (Optional)
    # --cropped_image_path path/to/your/cropped_image.jpg \ (Optional)
```

## Citation

If you use this codebase or the ideas from the associated paper, please cite the original work:

1. Title: Transformer guided Multimodal VQA model for Fruit recognitions
2. Authors: Dat Tran, Hoai Nam Vu
3. Affiliations: Thuyloi University, Hanoi, Viet Nam; YIRLODT Laboratory, Posts and Telecommunications Institute of Technology, Hanoi, Vietnam.