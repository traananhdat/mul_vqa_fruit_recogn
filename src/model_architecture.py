# src/model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, ViTConfig, BertConfig

# --- Default Configuration Values (can be overridden by passing a config dict to FruitVQAModel) ---
# These values are based on the paper's Table 1 and common practices.
DEFAULT_CONFIG = {
    'vit_model_name': "google/vit-base-patch16-224-in21k", # Pretrained ViT model
    'vit_pretrained': True,
    'vit_img_size': 224,
    'vit_patch_size': 16,
    'vit_hidden_size': 768,
    'vit_num_layers': 12,
    'vit_num_heads': 12,

    'bert_model_name': "bert-base-uncased", # Pretrained BERT model
    'bert_pretrained': True,
    'bert_hidden_size': 768,

    'visual_streams': 3, # Original, Segmented, Cropped
    'visual_encoder_input_dim_proj': 768, # Dimension after projecting concatenated stream features
    'visual_encoder_dim': 768,        # Hidden dimension of the Visual Transformer Encoder
    'visual_encoder_layers': 4,       # Number of layers in Visual Transformer Encoder (Paper mentions 12, can be heavy)
    'visual_encoder_heads': 8,        # Number of heads in Visual Transformer Encoder
    'visual_encoder_dropout': 0.1,

    'pca_input_dim': 768,             # Input to PCA module (output of Visual Encoder)
    'pca_output_dim': 768,            # Output of PCA module (transformed features)
                                      # If 768, PCA is for transformation, not dimensionality reduction for fusion input matching

    'fusion_input_concat_dim': 768 + 768, # pca_output_dim + bert_hidden_size
    'fusion_hidden_dim': 512,
    'fusion_dropout': 0.5,

    'num_question_filter_classes': 2, # e.g., agricultural vs. non-agricultural
    'num_vqa_answer_classes': 1000    # Placeholder: Update with actual number of answer classes from your dataset
}

# --- Model Components ---

class ImageStreamEncoder(nn.Module):
    """
    Encodes a single image stream using a Vision Transformer (ViT).
    """
    def __init__(self, model_name=DEFAULT_CONFIG['vit_model_name'], pretrained=DEFAULT_CONFIG['vit_pretrained'],
                 img_size=DEFAULT_CONFIG['vit_img_size'], patch_size=DEFAULT_CONFIG['vit_patch_size'],
                 hidden_size=DEFAULT_CONFIG['vit_hidden_size'], num_layers=DEFAULT_CONFIG['vit_num_layers'],
                 num_heads=DEFAULT_CONFIG['vit_num_heads']):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=True) # add_pooling_layer for pooler_output
        else:
            config = ViTConfig(hidden_size=hidden_size,
                               num_hidden_layers=num_layers,
                               num_attention_heads=num_heads,
                               image_size=img_size,
                               patch_size=patch_size,
                               # intermediate_size, hidden_act etc. can also be set
                               )
            self.vit = ViTModel(config, add_pooling_layer=True)

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Batch of images (batch_size, num_channels, height, width).
        Returns:
            torch.Tensor: Pooled output from ViT (batch_size, vit_hidden_size).
        """
        outputs = self.vit(pixel_values=images)
        return outputs.pooler_output

class VisualFeatureAggregator(nn.Module):
    """
    Aggregates features from multiple image streams, projects them,
    and then passes them through a Transformer Encoder.
    """
    def __init__(self, num_streams=DEFAULT_CONFIG['visual_streams'],
                 stream_feature_dim=DEFAULT_CONFIG['vit_hidden_size'],
                 projection_dim=DEFAULT_CONFIG['visual_encoder_input_dim_proj'],
                 encoder_dim=DEFAULT_CONFIG['visual_encoder_dim'],
                 num_encoder_layers=DEFAULT_CONFIG['visual_encoder_layers'],
                 num_encoder_heads=DEFAULT_CONFIG['visual_encoder_heads'],
                 dropout=DEFAULT_CONFIG['visual_encoder_dropout']):
        super().__init__()
        self.num_streams = num_streams
        self.concat_dim = num_streams * stream_feature_dim

        self.projection = nn.Linear(self.concat_dim, projection_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, # Should match projection_dim if projection_dim is input to encoder
            nhead=num_encoder_heads,
            dim_feedforward=encoder_dim * 4, # Common practice for feedforward size
            dropout=dropout,
            activation='gelu', # As per paper's Table 1 for "Visual Encoder" (BertLayer)
            batch_first=True    # Input format: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer_norm = nn.LayerNorm(encoder_dim) # Normalize the final output

        if projection_dim != encoder_dim:
            print(f"Warning: VisualFeatureAggregator projection_dim ({projection_dim}) != encoder_dim ({encoder_dim}). Ensure this is intended.")


    def forward(self, list_of_stream_features):
        """
        Args:
            list_of_stream_features (list of torch.Tensor): A list containing feature tensors
                                                            from each image stream. Each tensor
                                                            is (batch_size, stream_feature_dim).
        Returns:
            torch.Tensor: Encoded visual features (batch_size, encoder_dim).
        """
        if len(list_of_stream_features) != self.num_streams:
            raise ValueError(f"Expected {self.num_streams} feature streams, got {len(list_of_stream_features)}")

        concatenated_features = torch.cat(list_of_stream_features, dim=1)
        # concatenated_features: (batch_size, num_streams * stream_feature_dim)

        projected_features = self.projection(concatenated_features)
        # projected_features: (batch_size, projection_dim)

        # TransformerEncoder expects input of shape (batch_size, seq_len, feature_dim)
        # Our "sequence" here has length 1 (the aggregated projected feature vector)
        encoder_input = projected_features.unsqueeze(1) # (batch_size, 1, projection_dim)
        
        encoded_output = self.transformer_encoder(encoder_input) # (batch_size, 1, encoder_dim)
        encoded_output_squeezed = encoded_output.squeeze(1)    # (batch_size, encoder_dim)
        
        return self.output_layer_norm(encoded_output_squeezed)

class PCAModule(nn.Module):
    """
    Conceptual PCA module.
    In a real scenario, PCA components (mean and principal components) would be
    pre-fitted on the training dataset's features and loaded here.
    This version uses a linear layer to achieve the target output dimension,
    acting as a placeholder for a true PCA transformation or a learned projection.
    """
    def __init__(self, input_dim=DEFAULT_CONFIG['pca_input_dim'],
                 output_dim=DEFAULT_CONFIG['pca_output_dim'],
                 pre_fitted_mean=None, pre_fitted_components=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.uses_true_pca = False
        if pre_fitted_mean is not None and pre_fitted_components is not None:
            # This is where you would set up true PCA transformation
            # Ensure components are (input_dim, output_dim) and mean is (input_dim,)
            # And make them non-trainable buffers
            # self.register_buffer('mean', torch.tensor(pre_fitted_mean, dtype=torch.float32))
            # self.register_buffer('components', torch.tensor(pre_fitted_components, dtype=torch.float32)) # Shape (output_dim, input_dim) for sklearn or (input_dim, output_dim)
            # self.uses_true_pca = True
            # print(f"PCAModule: Configured with pre-fitted PCA (mean shape: {self.mean.shape}, components shape: {self.components.shape}).")
            # For now, still using linear as full PCA requires careful setup of components matrix
            print("PCAModule: Pre-fitted PCA components provided (conceptual). Using Linear for dimensionality transformation.")
            self.projection = nn.Linear(input_dim, output_dim)

        else:
            # If no pre-fitted PCA, use a linear layer. This layer will be trainable.
            # This can be seen as a learned linear transformation to the target dimension.
            print(f"PCAModule: No pre-fitted PCA. Using a trainable Linear layer from {input_dim} to {output_dim}.")
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (batch_size, input_dim).
        Returns:
            torch.Tensor: Transformed features (batch_size, output_dim).
        """
        # if self.uses_true_pca:
        #     x_centered = x - self.mean
        #     # Ensure components have shape (input_dim, output_dim) for x_centered @ components
        #     transformed_x = torch.matmul(x_centered, self.components.T[:, :self.output_dim]) # Or self.components directly if shaped (input_dim, output_dim)
        #     return transformed_x
        # else:
        return self.projection(x)

class TextEncoder(nn.Module):
    """
    Encodes text queries using a BERT model.
    """
    def __init__(self, model_name=DEFAULT_CONFIG['bert_model_name'], pretrained=DEFAULT_CONFIG['bert_pretrained']):
        super().__init__()
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig(hidden_size=DEFAULT_CONFIG['bert_hidden_size']) # Add other BertConfig params if needed
            self.bert = BertModel(config)
        # Optionally freeze BERT parameters
        # for param in self.bert.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids (torch.Tensor): Token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.
        Returns:
            torch.Tensor: Pooled output (CLS token representation) from BERT (batch_size, bert_hidden_size).
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        return outputs.pooler_output

class FusionModule(nn.Module):
    """
    Fuses visual and textual features.
    Paper Table 1: LayerNorm, Linear(1536,512), ReLU, Dropout(0.5)
    """
    def __init__(self, input_dim=DEFAULT_CONFIG['fusion_input_concat_dim'],
                 output_dim=DEFAULT_CONFIG['fusion_hidden_dim'],
                 dropout_rate=DEFAULT_CONFIG['fusion_dropout']):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features (torch.Tensor): PCA-transformed visual features (batch_size, pca_output_dim).
            text_features (torch.Tensor): BERT encoded text features (batch_size, bert_hidden_size).
        Returns:
            torch.Tensor: Fused representation (batch_size, fusion_hidden_dim).
        """
        # Ensure dimensions match the expected fusion_input_concat_dim
        if visual_features.shape[1] + text_features.shape[1] != self.layer_norm.normalized_shape[0]:
            raise ValueError(
                f"Dimension mismatch for fusion. Visual: {visual_features.shape[1]}, Text: {text_features.shape[1]}. "
                f"Expected sum: {self.layer_norm.normalized_shape[0]}"
            )

        combined_features = torch.cat([visual_features, text_features], dim=1)
        
        x = self.layer_norm(combined_features)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class QuestionFilterHead(nn.Module):
    """
    Output head for the domain-specific question filtering task.
    """
    def __init__(self, input_dim=DEFAULT_CONFIG['fusion_hidden_dim'],
                 num_classes=DEFAULT_CONFIG['num_question_filter_classes']):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, fused_features):
        return self.linear(fused_features)

class VQAAnswerHead(nn.Module):
    """
    Output head for the primary VQA task.
    """
    def __init__(self, input_dim=DEFAULT_CONFIG['fusion_hidden_dim'],
                 num_classes=DEFAULT_CONFIG['num_vqa_answer_classes']):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, fused_features):
        return self.linear(fused_features)

# --- Main VQA Model ---

class FruitVQAModel(nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        # Use provided config or default if None
        self.config = model_config if model_config is not None else DEFAULT_CONFIG.copy()

        # --- Image Processing Path ---
        # 1. Individual Image Stream Encoders
        self.image_encoder_original = ImageStreamEncoder(
            model_name=self.config['vit_model_name'], pretrained=self.config['vit_pretrained'],
            hidden_size=self.config['vit_hidden_size'] # Pass all relevant ViT params from config
        )
        self.image_encoder_segmented = ImageStreamEncoder(
            model_name=self.config['vit_model_name'], pretrained=self.config['vit_pretrained'],
            hidden_size=self.config['vit_hidden_size']
        )
        self.image_encoder_cropped = ImageStreamEncoder(
            model_name=self.config['vit_model_name'], pretrained=self.config['vit_pretrained'],
            hidden_size=self.config['vit_hidden_size']
        )

        # 2. Visual Feature Aggregator & Encoder
        self.visual_aggregator_encoder = VisualFeatureAggregator(
            num_streams=self.config['visual_streams'],
            stream_feature_dim=self.config['vit_hidden_size'],
            projection_dim=self.config['visual_encoder_input_dim_proj'], # Input to its internal Transformer encoder
            encoder_dim=self.config['visual_encoder_dim'], # Output dim of its internal Transformer encoder
            num_encoder_layers=self.config['visual_encoder_layers'],
            num_encoder_heads=self.config['visual_encoder_heads'],
            dropout=self.config['visual_encoder_dropout']
        )

        # 3. PCA Module
        self.pca_module = PCAModule(
            input_dim=self.config['visual_encoder_dim'], # Output of visual_aggregator_encoder
            output_dim=self.config['pca_output_dim']
            # To use real PCA, you'd pass pre_fitted_mean and pre_fitted_components here
        )

        # --- Text Processing Path ---
        self.text_encoder = TextEncoder(
            model_name=self.config['bert_model_name'],
            pretrained=self.config['bert_pretrained']
        )

        # --- Fusion and Output ---
        # Ensure fusion_input_concat_dim in config matches sum of pca_output_dim and bert_hidden_size
        expected_fusion_input_dim = self.config['pca_output_dim'] + self.config['bert_hidden_size']
        if self.config['fusion_input_concat_dim'] != expected_fusion_input_dim:
            print(f"Warning: config['fusion_input_concat_dim'] ({self.config['fusion_input_concat_dim']}) "
                  f"does not match pca_output_dim ({self.config['pca_output_dim']}) + "
                  f"bert_hidden_size ({self.config['bert_hidden_size']}). Using sum: {expected_fusion_input_dim}")
            actual_fusion_input_dim = expected_fusion_input_dim
        else:
            actual_fusion_input_dim = self.config['fusion_input_concat_dim']

        self.fusion_module = FusionModule(
            input_dim=actual_fusion_input_dim,
            output_dim=self.config['fusion_hidden_dim'],
            dropout_rate=self.config['fusion_dropout']
        )

        self.question_filter_head = QuestionFilterHead(
            input_dim=self.config['fusion_hidden_dim'],
            num_classes=self.config['num_question_filter_classes']
        )
        self.vqa_answer_head = VQAAnswerHead(
            input_dim=self.config['fusion_hidden_dim'],
            num_classes=self.config['num_vqa_answer_classes']
        )

    def forward(self, image_original, image_segmented, image_cropped,
                question_input_ids, question_attention_mask, question_token_type_ids=None):
        """
        Args:
            image_original (torch.Tensor): Batch of original images.
            image_segmented (torch.Tensor): Batch of segmented images.
            image_cropped (torch.Tensor): Batch of cropped images.
            question_input_ids (torch.Tensor): Token IDs for questions.
            question_attention_mask (torch.Tensor): Attention mask for questions.
            question_token_type_ids (torch.Tensor, optional): Token type IDs for questions.
        Returns:
            tuple: (filter_logits, vqa_answer_logits)
        """
        # 1. Image Features from three streams
        feat_original = self.image_encoder_original(image_original)
        feat_segmented = self.image_encoder_segmented(image_segmented)
        feat_cropped = self.image_encoder_cropped(image_cropped)

        # 2. Aggregate and Encode Visual Features
        # The VisualFeatureAggregator expects a list of features
        aggregated_visual_features = self.visual_aggregator_encoder(
            [feat_original, feat_segmented, feat_cropped]
        )

        # 3. PCA Transformation
        pca_transformed_visual_features = self.pca_module(aggregated_visual_features)

        # 4. Text Features
        text_features = self.text_encoder(
            input_ids=question_input_ids,
            attention_mask=question_attention_mask,
            token_type_ids=question_token_type_ids
        )

        # 5. Fusion
        fused_representation = self.fusion_module(pca_transformed_visual_features, text_features)

        # 6. Output Heads
        filter_logits = self.question_filter_head(fused_representation)
        vqa_answer_logits = self.vqa_answer_head(fused_representation)

        return filter_logits, vqa_answer_logits


if __name__ == '__main__':
    # This part is for a quick test if the script is run directly.
    # It's better to do more thorough testing in a notebook like 03_model_prototyping.ipynb

    print("Model Architecture Script (`model_architecture.py`)")
    print("Testing model instantiation with default config...")

    # Use a copy of default config to avoid modifying it globally if tests change it
    current_config = DEFAULT_CONFIG.copy()
    # IMPORTANT: For this test, NUM_VQA_ANSWER_CLASSES is a placeholder.
    # It should be set based on your actual dataset's answer vocabulary size.
    # For example, if your build_answer_vocab function (from data_loader.py)
    # results in 1002 answer classes (top 1000 + UNK + PAD), set it here.
    current_config['num_vqa_answer_classes'] = 1000 # Example

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = FruitVQAModel(model_config=current_config).to(device)
        model.eval() # Set to eval mode for testing
        print("FruitVQAModel instantiated successfully.")

        # Create dummy inputs
        batch_size = 2
        img_size = current_config['vit_img_size']
        max_token_len = 128 # Should match preprocessing

        dummy_img_orig = torch.randn(batch_size, 3, img_size, img_size).to(device)
        dummy_img_seg = torch.randn(batch_size, 3, img_size, img_size).to(device)
        dummy_img_crop = torch.randn(batch_size, 3, img_size, img_size).to(device)

        dummy_q_ids = torch.randint(0, 1000, (batch_size, max_token_len)).to(device)
        dummy_q_mask = torch.ones(batch_size, max_token_len, dtype=torch.long).to(device)
        dummy_q_type_ids = torch.zeros(batch_size, max_token_len, dtype=torch.long).to(device)

        with torch.no_grad():
            filter_logits, vqa_logits = model(dummy_img_orig, dummy_img_seg, dummy_img_crop,
                                              dummy_q_ids, dummy_q_mask, dummy_q_type_ids)

        print("\n--- Dummy Forward Pass Output Shapes ---")
        print(f"Filter Logits Shape: {filter_logits.shape}") # Expected: (batch_size, num_question_filter_classes)
        print(f"VQA Logits Shape:    {vqa_logits.shape}")    # Expected: (batch_size, num_vqa_answer_classes)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters in FruitVQAModel: {num_params:,}")

    except ImportError as e:
        print(f"ImportError: {e}. Make sure 'transformers' and 'torch' are installed.")
        print("You might need to run: pip install torch torchvision transformers")
    except Exception as e:
        print(f"An error occurred during model testing: {e}")
        import traceback
        traceback.print_exc()