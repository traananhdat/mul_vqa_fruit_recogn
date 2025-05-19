# tests/test_model_architecture.py

import unittest
import torch
import torch.nn as nn

# Assuming 'src' is in PYTHONPATH or tests are run from project root
from src.model_architecture import (
    ImageStreamEncoder,
    VisualFeatureAggregator,
    PCAModule,
    TextEncoder,
    FusionModule,
    QuestionFilterHead,
    VQAAnswerHead,
    FruitVQAModel,
    DEFAULT_CONFIG # Use this as a base for our test config
)

class TestModelArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning model architecture tests on device: {cls.device}")

        # Create a test configuration, possibly overriding some defaults for speed/simplicity in tests
        # For shape testing, it's good to keep most dimensions as intended by DEFAULT_CONFIG
        # but ensure pretrained=False for ViT and BERT to avoid downloads.
        cls.test_config = DEFAULT_CONFIG.copy()
        cls.test_config.update({
            'vit_pretrained': False, # Avoid downloading ViT weights during unit tests
            'bert_pretrained': False, # Avoid downloading BERT weights during unit tests
            'visual_encoder_layers': 2, # Reduce layers for faster test instantiation if needed
            'num_vqa_answer_classes': 10, # A smaller number for test VQA head
            # PCAModule uses a Linear layer if no pre-fitted components, which is fine for testing flow
        })
        cls.batch_size = 2
        cls.img_size = cls.test_config['vit_img_size']
        cls.max_token_length = cls.test_config['max_token_length']


    def test_01_image_stream_encoder(self):
        print("Testing ImageStreamEncoder...")
        model = ImageStreamEncoder(
            pretrained=self.test_config['vit_pretrained'],
            hidden_size=self.test_config['vit_hidden_size'],
            img_size=self.img_size,
            patch_size=self.test_config['vit_patch_size']
        ).to(self.device)
        model.eval()
        dummy_image = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        output = model(dummy_image)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['vit_hidden_size']))

    def test_02_visual_feature_aggregator(self):
        print("Testing VisualFeatureAggregator...")
        model = VisualFeatureAggregator(
            num_streams=self.test_config['visual_streams'],
            stream_feature_dim=self.test_config['vit_hidden_size'],
            projection_dim=self.test_config['visual_encoder_input_dim_proj'],
            encoder_dim=self.test_config['visual_encoder_dim'],
            num_encoder_layers=self.test_config['visual_encoder_layers'],
            num_encoder_heads=self.test_config['visual_encoder_heads'],
            dropout=self.test_config['visual_encoder_dropout']
        ).to(self.device)
        model.eval()
        
        dummy_stream_features = [
            torch.randn(self.batch_size, self.test_config['vit_hidden_size']).to(self.device)
            for _ in range(self.test_config['visual_streams'])
        ]
        output = model(dummy_stream_features)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['visual_encoder_dim']))

    def test_03_pca_module(self):
        print("Testing PCAModule...")
        # PCAModule in model_architecture.py uses a Linear layer if no pre-fitted components
        model = PCAModule(
            input_dim=self.test_config['pca_input_dim'],
            output_dim=self.test_config['pca_output_dim']
            # pre_fitted_mean and pre_fitted_components would be None by default here
        ).to(self.device)
        model.eval()
        dummy_features = torch.randn(self.batch_size, self.test_config['pca_input_dim']).to(self.device)
        output = model(dummy_features)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['pca_output_dim']))

    def test_04_text_encoder(self):
        print("Testing TextEncoder...")
        model = TextEncoder(
            model_name=self.test_config['bert_model_name'],
            pretrained=self.test_config['bert_pretrained']
        ).to(self.device)
        model.eval()
        dummy_input_ids = torch.randint(0, 1000, (self.batch_size, self.max_token_length)).to(self.device)
        dummy_attention_mask = torch.ones(self.batch_size, self.max_token_length, dtype=torch.long).to(self.device)
        output = model(dummy_input_ids, dummy_attention_mask)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['bert_hidden_size']))

    def test_05_fusion_module(self):
        print("Testing FusionModule...")
        # Calculate expected input_dim for fusion based on config
        fusion_input_dim = self.test_config['pca_output_dim'] + self.test_config['bert_hidden_size']
        
        model = FusionModule(
            input_dim=fusion_input_dim,
            output_dim=self.test_config['fusion_hidden_dim'],
            dropout_rate=self.test_config['fusion_dropout']
        ).to(self.device)
        model.eval()
        
        dummy_visual_features = torch.randn(self.batch_size, self.test_config['pca_output_dim']).to(self.device)
        dummy_text_features = torch.randn(self.batch_size, self.test_config['bert_hidden_size']).to(self.device)
        output = model(dummy_visual_features, dummy_text_features)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['fusion_hidden_dim']))

        # Test dimension mismatch
        wrong_dim_visual_features = torch.randn(self.batch_size, self.test_config['pca_output_dim'] -10).to(self.device)
        with self.assertRaises(ValueError): # Or RuntimeError depending on how nn.LayerNorm or torch.cat behaves
             model(wrong_dim_visual_features, dummy_text_features)


    def test_06_question_filter_head(self):
        print("Testing QuestionFilterHead...")
        model = QuestionFilterHead(
            input_dim=self.test_config['fusion_hidden_dim'],
            num_classes=self.test_config['num_question_filter_classes']
        ).to(self.device)
        model.eval()
        dummy_fused_features = torch.randn(self.batch_size, self.test_config['fusion_hidden_dim']).to(self.device)
        output = model(dummy_fused_features)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['num_question_filter_classes']))

    def test_07_vqa_answer_head(self):
        print("Testing VQAAnswerHead...")
        model = VQAAnswerHead(
            input_dim=self.test_config['fusion_hidden_dim'],
            num_classes=self.test_config['num_vqa_answer_classes'] # Using test config value
        ).to(self.device)
        model.eval()
        dummy_fused_features = torch.randn(self.batch_size, self.test_config['fusion_hidden_dim']).to(self.device)
        output = model(dummy_fused_features)
        self.assertEqual(output.shape, (self.batch_size, self.test_config['num_vqa_answer_classes']))

    def test_08_fruit_vqa_model_instantiation_and_forward_pass(self):
        print("Testing FruitVQAModel instantiation and forward pass...")
        try:
            model = FruitVQAModel(model_config=self.test_config).to(self.device)
            model.eval()
        except Exception as e:
            self.fail(f"FruitVQAModel instantiation failed: {e}")

        dummy_img_orig = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        dummy_img_seg = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        dummy_img_crop = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        dummy_q_ids = torch.randint(0, 1000, (self.batch_size, self.max_token_length)).to(self.device)
        dummy_q_mask = torch.ones(self.batch_size, self.max_token_length, dtype=torch.long).to(self.device)
        dummy_q_type_ids = torch.zeros(self.batch_size, self.max_token_length, dtype=torch.long).to(self.device)

        try:
            with torch.no_grad():
                filter_logits, vqa_logits = model(
                    dummy_img_orig, dummy_img_seg, dummy_img_crop,
                    dummy_q_ids, dummy_q_mask, dummy_q_type_ids
                )
        except Exception as e:
            self.fail(f"FruitVQAModel forward pass failed: {e}")

        self.assertEqual(filter_logits.shape, (self.batch_size, self.test_config['num_question_filter_classes']))
        self.assertEqual(vqa_logits.shape, (self.batch_size, self.test_config['num_vqa_answer_classes']))
        print("FruitVQAModel forward pass successful.")

    def test_09_model_to_cuda(self):
        print("Testing model movement to CUDA (if available)...")
        if torch.cuda.is_available():
            model = FruitVQAModel(model_config=self.test_config)
            model.to(torch.device("cuda"))
            # Check if a parameter is on CUDA
            self.assertTrue(next(model.parameters()).is_cuda, "Model parameter not on CUDA device.")
            print("Model successfully moved to CUDA.")
        else:
            self.skipTest("CUDA not available, skipping CUDA specific test.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)