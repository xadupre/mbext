# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

_MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"


def _make_small_phi4mm_config():
    """Create a small ``Phi4MultimodalConfig`` for unit tests.

    Uses ``image_size=56`` (56/14=4 patches per side) and ``num_hidden_layers=4``
    for the vision encoder so the test completes quickly.  Text decoder uses 1
    hidden layer.  ``feature_layer=-2`` means we extract features after the
    second-to-last layer (layer index 2).
    """
    from transformers import Phi4MultimodalAudioConfig, Phi4MultimodalConfig, Phi4MultimodalVisionConfig

    vision_cfg = Phi4MultimodalVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=56,
        crop_size=56,
        patch_size=14,
        feature_layer=-2,
        layer_norm_eps=1e-6,
    )
    audio_cfg = Phi4MultimodalAudioConfig()
    cfg = Phi4MultimodalConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=210000,
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        rms_norm_eps=1e-5,
        vision_config=vision_cfg,
        audio_config=audio_cfg,
    )
    cfg.architectures = ["Phi4MultimodalForCausalLM"]
    return cfg


@requires_transformers("5")
class TestPhi4Multimodal(ExtTestCase):
    # ------------------------------------------------------------------ #
    #  Vision encoder tests                                               #
    # ------------------------------------------------------------------ #

    @hide_stdout()
    def test_vision_encoder_fp32_cpu_builds_and_runs(self):
        """Vision encoder exports to ONNX and produces correct output shape."""
        import torch
        from transformers import Phi4MultimodalForCausalLM

        from modelbuilder.builder import set_io_dtype, set_onnx_dtype
        from modelbuilder.builders.phi import Phi4MultimodalVisionEncoderModel

        cfg = _make_small_phi4mm_config()
        torch.manual_seed(42)
        model = Phi4MultimodalForCausalLM(cfg)
        model.eval()

        model_dir = self.get_model_dir("test_phi4mm_vis_fp32")
        model.save_pretrained(model_dir)

        cache_dir = self.get_model_dir("test_phi4mm_vis_fp32_cache")
        extra_options = {}
        io_dtype = set_io_dtype("fp32", "cpu", extra_options)
        onnx_dtype = set_onnx_dtype("fp32", extra_options)

        encoder = Phi4MultimodalVisionEncoderModel(cfg, io_dtype, onnx_dtype, "cpu", cache_dir, extra_options)
        encoder.make_model(model_dir)

        out_dir, _ = self.get_dirs("test_phi4mm_vis_fp32")
        encoder.save_model(out_dir)

        vis_onnx = os.path.join(out_dir, "vision_encoder.onnx")
        self.assertExists(vis_onnx)

        # Run the ONNX model.
        sess = self.check_ort(vis_onnx)
        pixel_values = np.zeros((2, 3, 56, 56), dtype=np.float32)
        outputs = sess.run(None, {"pixel_values": pixel_values})
        self.assertIsNotNone(outputs[0])

        # n_cs = 2, n_image_tokens = 2*3 + 1 + 2*3 = 13
        self.assertEqual(outputs[0].shape, (encoder.n_image_tokens, cfg.hidden_size))

    @hide_stdout()
    def test_vision_encoder_fp32_numerical_accuracy(self):
        """ONNX vision encoder output closely matches the PyTorch reference."""
        import torch
        import torch.nn.functional as F
        from transformers import Phi4MultimodalForCausalLM

        from modelbuilder.builder import set_io_dtype, set_onnx_dtype
        from modelbuilder.builders.phi import Phi4MultimodalVisionEncoderModel

        cfg = _make_small_phi4mm_config()
        torch.manual_seed(42)
        model = Phi4MultimodalForCausalLM(cfg)
        model.eval()

        model_dir = self.get_model_dir("test_phi4mm_vis_accuracy")
        model.save_pretrained(model_dir)

        cache_dir = self.get_model_dir("test_phi4mm_vis_accuracy_cache")
        extra_options = {}
        io_dtype = set_io_dtype("fp32", "cpu", extra_options)
        onnx_dtype = set_onnx_dtype("fp32", extra_options)

        encoder = Phi4MultimodalVisionEncoderModel(cfg, io_dtype, onnx_dtype, "cpu", cache_dir, extra_options)
        encoder.make_model(model_dir)
        out_dir, _ = self.get_dirs("test_phi4mm_vis_accuracy")
        encoder.save_model(out_dir)

        ie = model.model.embed_tokens_extend.image_embed
        vis_model = ie.img_processor
        n_cs = encoder.n_compressed_per_side  # 2
        n_cp = encoder.n_compressed_patches  # 4

        sess = self.check_ort(os.path.join(out_dir, "vision_encoder.onnx"))

        torch.manual_seed(0)
        pixel_values = torch.randn(2, 3, 56, 56)

        # Reference PyTorch output.
        with torch.no_grad():
            pat_mask = torch.ones(2, 4, 4, dtype=torch.bool)
            hs = vis_model(pixel_values, patch_attention_mask=pat_mask, output_hidden_states=True)
            feat = hs.hidden_states[-2]  # [2, n_patches, vis_hidden]
            pf = feat.view(2, n_cs * 2, n_cs * 2, cfg.vision_config.hidden_size).permute(0, 3, 1, 2)
            pf = torch.nn.functional.avg_pool2d(pf, 2, 2)
            pf = pf.permute(0, 2, 3, 1).view(2, n_cp, cfg.vision_config.hidden_size)

            global_img = pf[0:1]
            sub_img = pf[1:2]
            sub_ext = ie.sub_img_feature_extensor.repeat(1, n_cs, 1, 1)
            global_img = global_img.reshape(1, n_cs, n_cs, -1)
            global_img = torch.cat([global_img, sub_ext], dim=2).reshape(1, -1, cfg.vision_config.hidden_size)
            sub_img = sub_img.reshape(1, n_cs, n_cs, -1)
            sub_img = torch.cat([sub_img, sub_ext], dim=2).reshape(1, -1, cfg.vision_config.hidden_size)
            merged = torch.cat([sub_img, ie.global_img_feature_extensor, global_img], dim=1)
            out = ie.img_projection_up(merged)
            out = F.gelu(out, approximate="tanh")
            out = ie.img_projection_down(out)
            pt_result = out.squeeze(0).numpy()

        onnx_result = sess.run(None, {"pixel_values": pixel_values.numpy()})[0]
        self.assertEqual(pt_result.shape, onnx_result.shape)
        self.assertLess(float(np.max(np.abs(pt_result - onnx_result))), 0.01)

    # ------------------------------------------------------------------ #
    #  Full multimodal pipeline test                                      #
    # ------------------------------------------------------------------ #

    def common_phi4mm_conditional_generation(self, precision, provider):
        """Build and validate the full Phi4Multimodal ONNX pipeline.

        Verifies that ``create_model`` (with ``multimodal=true``) produces:

        * ``vision_encoder.onnx`` — vision tower + compression + projection.
        * ``embedding.onnx``      — token embedding + ScatterND image scatter.
        * ``model.onnx``          — text decoder (``inputs_embeds`` → logits).
        * ``genai_config.json``   — ``phi3v`` config with vision+embedding sections.

        Also runs ORT forward passes to confirm correct output shapes.  The
        vision encoder pixel values and all model I/O use the dtype that
        corresponds to the model precision (int4/cpu → float32, fp16 →
        float16).
        """
        import torch
        from transformers import Phi4MultimodalForCausalLM

        from modelbuilder.builder import create_model

        cfg = _make_small_phi4mm_config()
        torch.manual_seed(42)
        model = Phi4MultimodalForCausalLM(cfg)
        model.eval()

        basename = f"test_phi4mm_cond_gen_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=1,
            multimodal=True,
        )

        # --- File existence ---
        vision_onnx = os.path.join(output_dir, "vision_encoder.onnx")
        embed_onnx = os.path.join(output_dir, "embedding.onnx")
        text_onnx = os.path.join(output_dir, "model.onnx")
        for path in (vision_onnx, embed_onnx, text_onnx):
            self.assertExists(path)

        # --- genai_config.json structure ---
        self.check_phi3v_genai_config(output_dir)

        # --- Vision encoder ORT forward pass ---
        # The vision encoder pixel_values dtype follows io_dtype (float16 for
        # fp16 models, float32 for fp32/int4 models).
        # n_cs = 2 → n_image_tokens = 2*3 + 1 + 2*3 = 13
        np_dtype = self.get_input_np_dtype(precision)
        pixel_values = np.zeros((2, 3, 56, 56), dtype=np_dtype)
        vis_features = self.run_vision_encoder_ort_check(vision_onnx, pixel_values, (13, cfg.hidden_size), provider=provider)

        # --- Text decoder ORT forward pass ---
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_ids).numpy().astype(np_dtype)

        self.run_inputs_embeds_decoder_check(
            text_onnx,
            inputs_embeds,
            num_hidden_layers=cfg.num_hidden_layers,
            num_key_value_heads=cfg.num_key_value_heads,
            head_size=cfg.hidden_size // cfg.num_attention_heads,
            precision=precision,
            provider=provider,
        )

        # --- Embedding model ORT forward pass ---
        embed_sess = self.check_ort(embed_onnx, provider=provider)
        image_token_id = cfg.vision_config.image_token_id
        n_image_tokens = 13
        input_ids_with_img = np.array([[image_token_id] * n_image_tokens], dtype=np.int64)
        # image_features dtype must match the embedding model's expected I/O dtype
        image_features = vis_features.astype(np_dtype)
        embed_out = embed_sess.run(None, {"input_ids": input_ids_with_img, "image_features": image_features})
        self.assertIsNotNone(embed_out[0])
        self.assertEqual(embed_out[0].shape, (batch_size, n_image_tokens, cfg.hidden_size))

    @hide_stdout()
    def test_phi4_multimodal_conditional_generation_fp32_cpu_random_weights(self):
        """fp32 / CPU pipeline — see :meth:`common_phi4mm_conditional_generation`."""
        self.common_phi4mm_conditional_generation("fp32", "cpu")

    @hide_stdout()
    def test_phi4_multimodal_conditional_generation_fp16_cpu_random_weights(self):
        """fp16 / CPU pipeline — see :meth:`common_phi4mm_conditional_generation`."""
        self.common_phi4mm_conditional_generation("fp16", "cpu")

    @hide_stdout()
    def test_phi4_multimodal_conditional_generation_int4_cpu_random_weights(self):
        """int4 / CPU pipeline — see :meth:`common_phi4mm_conditional_generation`."""
        self.common_phi4mm_conditional_generation("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_phi4_multimodal_conditional_generation_fp16_cuda_random_weights(self):
        """fp16 / CUDA pipeline — see :meth:`common_phi4mm_conditional_generation`."""
        self.common_phi4mm_conditional_generation("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
