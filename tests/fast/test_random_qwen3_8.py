# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_transformers

QWEN3_8_MODEL_NAME = "Qwen/Qwen3.8-27B"


def _make_qwen3_8_config():
    """Create a tiny offline configuration with the Qwen3.8 architecture."""
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [2, 3, 3],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
        "rope_type": "default",
    }
    text_config = Qwen3_5TextConfig(
        attention_bias=False,
        attn_output_gate=True,
        bos_token_id=1,
        eos_token_id=2,
        full_attention_interval=4,
        head_dim=64,
        hidden_size=128,
        intermediate_size=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=3,
        linear_value_head_dim=16,
        max_position_embeddings=256,
        num_attention_heads=2,
        num_hidden_layers=4,
        num_key_value_heads=1,
        output_gate_type="swish",
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        rope_parameters=rope_parameters,
        vocab_size=128,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_act="gelu_pytorch_tanh",
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=128,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    config = Qwen3_5Config(
        image_token_id=120,
        text_config=text_config,
        tie_word_embeddings=False,
        video_token_id=121,
        vision_config=vision_config,
        vision_end_token_id=123,
        vision_start_token_id=122,
    )
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomQwen3_8(ExtTestCase):
    def _build_multimodal_model(self):
        import torch
        from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5ForConditionalGeneration

        from modelbuilder.builder import create_model

        config = _make_qwen3_8_config()
        prefix = "test_random_qwen3_8_multimodal"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)

        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            ),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            ),
        )
        processor.save_pretrained(model_dir)

        create_model(
            model_name=QWEN3_8_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            multimodal=True,
        )
        return config, model, output_dir

    @hide_stdout()
    def test_qwen3_8_dispatches_to_qwen3_5_builder(self):
        from onnx_light.onnx import TensorProto

        from modelbuilder.builders.qwen import Qwen35TextModel

        config = _make_qwen3_8_config()
        model = Qwen35TextModel(config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", "", {})
        self.assertIsInstance(model, Qwen35TextModel)
        self.assertEqual(model.layer_types.count("linear_attention"), 3)
        self.assertEqual(model.layer_types.count("full_attention"), 1)

    @hide_stdout()
    def test_qwen3_8_multimodal_conversion_and_parity(self):
        import torch

        config, model, output_dir = self._build_multimodal_model()
        expected_files = {"embedding.onnx", "genai_config.json", "model.onnx", "processor_config.json", "vision.onnx"}
        self.assertTrue(expected_files.issubset(set(os.listdir(output_dir))), os.listdir(output_dir))

        with open(os.path.join(output_dir, "genai_config.json")) as file:
            genai_config = json.load(file)
        self.assertEqual(genai_config["model"]["type"], "qwen3_5")
        self.assertEqual(genai_config["model"]["vision"]["filename"], "vision.onnx")
        self.assertEqual(genai_config["model"]["vision"]["inputs"], {"image_grid_thw": "image_grid_thw", "pixel_values": "pixel_values"})
        self.assertEqual(genai_config["model"]["embedding"]["filename"], "embedding.onnx")
        self.assertEqual(genai_config["model"]["image_token_id"], 120)
        self.assertEqual(genai_config["model"]["video_token_id"], 121)
        decoder = genai_config["model"]["decoder"]
        self.assertEqual(decoder["num_hidden_layers"], 1)
        self.assertEqual(decoder["inputs"]["past_conv_names"], "past_key_values.%d.conv_state")
        self.assertEqual(decoder["inputs"]["past_recurrent_names"], "past_key_values.%d.recurrent_state")

        vision_session = self.check_ort(os.path.join(output_dir, "vision.onnx"))
        embedding_session = self.check_ort(os.path.join(output_dir, "embedding.onnx"))
        vision_config = config.vision_config
        input_width = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2

        for temporal, height, width in ((1, 4, 4), (1, 4, 6), (2, 4, 4)):
            grid = torch.tensor([[temporal, height, width]], dtype=torch.int64)
            torch.manual_seed(temporal * 100 + height * 10 + width)
            pixel_values = torch.randn(temporal * height * width, input_width)
            with torch.no_grad():
                expected = model.model.visual(pixel_values, grid).pooler_output.numpy()
            (actual,) = vision_session.run(None, {"image_grid_thw": grid.numpy(), "pixel_values": pixel_values.numpy()})
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)

        image_features = actual
        input_ids = np.array([[5] + [config.image_token_id] * image_features.shape[0] + [6]], dtype=np.int64)
        (inputs_embeds,) = embedding_session.run(None, {"image_features": image_features, "input_ids": input_ids})
        with torch.no_grad():
            expected_embeds = model.model.language_model.embed_tokens(torch.from_numpy(input_ids)).numpy()
        expected_embeds[0, 1:-1] = image_features
        np.testing.assert_allclose(inputs_embeds, expected_embeds, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
