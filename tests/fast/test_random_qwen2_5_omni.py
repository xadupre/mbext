# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

QWEN2_5_OMNI_MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"


def _make_qwen25_omni_thinker_config(num_hidden_layers=1):
    """Return a minimal ``Qwen2_5OmniThinkerConfig`` for offline unit tests.

    The config uses small model dimensions so that the test runs quickly
    without downloading any model weights.  The ``architectures`` attribute
    is set to ``["Qwen2_5OmniForConditionalGeneration"]`` so that
    :func:`modelbuilder.builder.create_model` dispatches to
    :class:`~modelbuilder.builders.qwen.Qwen25OmniThinkerModel`.

    Qwen2.5-Omni uses multimodal RoPE (mRoPE) with ``mrope_section`` stored
    in ``rope_parameters`` rather than ``rope_scaling``.  The minimal audio
    and vision tower configs are kept tiny to avoid OOM in CI.
    """
    from transformers import Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniThinkerConfig, Qwen2_5OmniVisionEncoderConfig

    # Minimal audio encoder: single layer, very small hidden size.
    # output_dim must equal text hidden_size.
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=32, output_dim=256
    )

    # Minimal vision encoder: single layer, very small hidden size.
    # out_hidden_size must equal text hidden_size.
    vision_config = Qwen2_5OmniVisionEncoderConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256)

    config = Qwen2_5OmniThinkerConfig(audio_config=audio_config, vision_config=vision_config)

    # Override text config with small dimensions.
    # head_size = hidden_size // num_attention_heads = 256 // 4 = 64
    # sum(mrope_section) * 2 must equal head_size:
    #   sum([8, 12, 12]) = 32, 32 * 2 = 64 == head_size ✓
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.num_hidden_layers = num_hidden_layers
    config.text_config.layer_types = ["full_attention"] * num_hidden_layers
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.vocab_size = 32000
    config.text_config.bos_token_id = 1
    config.text_config.eos_token_id = 2
    config.text_config.rms_norm_eps = 1e-6
    config.text_config.max_position_embeddings = 2048
    config.text_config.hidden_act = "silu"
    # mRoPE parameters: sections must satisfy sum(mrope_section) * 2 == head_size
    config.text_config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [8, 12, 12]}

    # Set architecture to trigger the Qwen2_5OmniForConditionalGeneration
    # dispatch path in builder.py.
    config.architectures = ["Qwen2_5OmniForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomQwen25Omni(ExtTestCase):
    def common_fast_qwen25omni_random_weights(self, precision, provider):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        num_hidden_layers = 1
        config = _make_qwen25_omni_thinker_config(num_hidden_layers)

        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        head_size = config.text_config.hidden_size // config.text_config.num_attention_heads
        self.run_vl_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            basename=f"test_discrepancies_qwen25omni_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.text_config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            pt_mode="inputs_embeds",
        )

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_fp32_cpu(self):
        self.common_fast_qwen25omni_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_fp16_cpu(self):
        self.common_fast_qwen25omni_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_int4_cpu(self):
        self.common_fast_qwen25omni_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25omni_fp16_cuda(self):
        self.common_fast_qwen25omni_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25omni_bf16_cuda(self):
        self.common_fast_qwen25omni_random_weights("bf16", "cuda")

    @unittest.skip(
        "RuntimeError: No execution providers were provided or selected – "
        "ORT-GenAI 0.13.x initialises Qwen2_5_VL as a full multimodal pipeline "
        "and cannot load the text-only thinker ONNX on CPU. "
        "Un-skip when ORT-GenAI supports text-only mRoPE decoder inference."
    )
    @hide_stdout()
    @requires_genai()
    def test_qwen25omni_fp32_cpu_genai_generate(self):
        """Verify that ``onnxruntime-genai`` can load and run a Qwen2.5-Omni thinker model.

        This test builds a minimal random-weight Qwen2.5-Omni thinker ONNX model
        (fp32 / CPU) and verifies that ORT-genai can load and generate from it.
        PyTorch parity is not checked here because the mRoPE position-id computation
        differs between PyTorch (internal ``get_rope_index``) and ORT-genai; what we
        validate instead is that the genai pipeline executes without error.
        """
        import torch

        from modelbuilder.builder import create_model

        prefix = "test_qwen25omni_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = _make_qwen25_omni_thinker_config(num_hidden_layers)

        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # Pass model=None to skip PyTorch comparison: mRoPE position-id handling
        # differs between PyTorch (computed internally by get_rope_index) and
        # ORT-genai (computed from the genai pipeline).  The test still verifies
        # that model.onnx and genai_config.json are valid and that ORT-genai can
        # execute a generation loop without error.
        self.run_genai_generation_test(
            output_dir, model=None, vocab_size=config.text_config.vocab_size, eos_token_id=config.text_config.eos_token_id
        )


@requires_transformers("5")
class TestRandomQwen25OmniVision(ExtTestCase):
    """Tests for the Qwen2.5-Omni multimodal pipeline (vision + audio encoders +
    embedding model + thinker text decoder).

    The vision encoder takes:

    * ``pixel_values`` [n_patches, in_feat_dim] – flat image patches from the image processor.
    * ``rotary_pos_emb`` [n_patches, head_dim // 2] – 2-D RoPE frequencies (float32).

    The audio encoder takes:

    * ``input_features`` [num_mel_bins, n_frames] – mel-spectrogram for a single audio item.

    The embedding model replaces ``image_token_id`` and ``audio_token_id`` placeholders
    in ``input_ids`` with the corresponding encoder outputs.
    """

    def common_qwen25omni_conditional_generation(self, precision, provider):
        """Build and validate the full Qwen2.5-Omni multimodal ONNX pipeline.

        Verifies that:

        * ``vision_encoder.onnx``, ``audio_encoder.onnx``, ``embedding.onnx``,
          and ``model.onnx`` are written.
        * ``genai_config.json`` has the ``phi3v`` type with ``vision``, ``audio``,
          and ``embedding`` sections.
        * The vision encoder produces output of the correct shape and matches the
          PyTorch reference.
        * The audio encoder produces output of the correct shape and matches the
          PyTorch reference (single-chunk audio, within one window).
        * The text decoder produces logits when fed ``inputs_embeds``.
        """
        import json

        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            PreTrainedTokenizerFast,
            Qwen2_5OmniAudioEncoderConfig,
            Qwen2_5OmniThinkerConfig,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniVisionEncoderConfig,
        )

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        spatial_merge_size = 2

        audio_config = Qwen2_5OmniAudioEncoderConfig(
            encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=32, output_dim=256
        )
        vision_config = Qwen2_5OmniVisionEncoderConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256)
        config = Qwen2_5OmniThinkerConfig(audio_config=audio_config, vision_config=vision_config)
        config.text_config.hidden_size = 256
        config.text_config.intermediate_size = 512
        config.text_config.num_hidden_layers = num_hidden_layers
        config.text_config.layer_types = ["full_attention"] * num_hidden_layers
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 2
        config.text_config.vocab_size = 32000
        config.text_config.bos_token_id = 1
        config.text_config.eos_token_id = 2
        config.text_config.rms_norm_eps = 1e-6
        config.text_config.max_position_embeddings = 2048
        config.text_config.hidden_act = "silu"
        config.text_config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [8, 12, 12]}
        config.architectures = ["Qwen2_5OmniForConditionalGeneration"]

        prefix = f"test_qwen25omni_conditional_generation_{precision}_{provider}"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)

        torch.manual_seed(42)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
            multimodal=True,
        )

        # --- Verify all ONNX artefacts exist ---
        vision_onnx_path = os.path.join(output_dir, "vision_encoder.onnx")
        audio_onnx_path = os.path.join(output_dir, "audio_encoder.onnx")
        embedding_onnx_path = os.path.join(output_dir, "embedding.onnx")
        text_onnx_path = os.path.join(output_dir, "model.onnx")
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        for p in [vision_onnx_path, audio_onnx_path, embedding_onnx_path, text_onnx_path, genai_config_path]:
            self.assertExists(p)

        # --- Verify genai_config.json ---
        with open(genai_config_path) as f:
            genai_config = json.load(f)
        self.assertEqual(genai_config["model"]["type"], "phi3v")
        self.assertIn("vision", genai_config["model"])
        self.assertEqual(genai_config["model"]["vision"]["filename"], "vision_encoder.onnx")
        self.assertIn("audio", genai_config["model"])
        self.assertEqual(genai_config["model"]["audio"]["filename"], "audio_encoder.onnx")
        self.assertIn("embedding", genai_config["model"])
        self.assertEqual(genai_config["model"]["embedding"]["filename"], "embedding.onnx")

        np_dtype = np.float16 if precision == "fp16" else np.float32

        log_data = dict(
            precision=precision,
            model_id=QWEN2_5_OMNI_MODEL_NAME,
            experiment="multimodal",
            provider=provider,
            test=prefix,
            input_type="vision+audio+text",
            kind="fast",
        )

        # --- Run vision encoder and compare to PyTorch ---
        vc = config.vision_config
        in_feat_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # 2×2 LLM grid (1 temporal frame, 4×4 vision patches = 16 patches)
        t, h, w = 1, 2 * spatial_merge_size, 2 * spatial_merge_size
        n_patches = t * h * w
        grid_thw = torch.tensor([[t, h, w]], dtype=torch.int32)
        n_merged = n_patches // (spatial_merge_size**2)

        torch.manual_seed(0)
        pixel_values = torch.randn(n_patches, in_feat_dim)

        # Compute PyTorch reference (always in float32).
        with torch.no_grad():
            rotary_pos_emb = model.visual.rot_pos_emb(grid_thw)
            pt_vis_out = model.visual(hidden_states=pixel_values, grid_thw=grid_thw)
            pt_vis_features = pt_vis_out.pooler_output.numpy().astype(np.float32)

        vision_sess = self.check_ort(vision_onnx_path)
        vision_out = vision_sess.run(
            None, {"pixel_values": pixel_values.numpy().astype(np.float32), "rotary_pos_emb": rotary_pos_emb.numpy().astype(np.float32)}
        )
        self.assertIsNotNone(vision_out[0])
        self.assertEqual(vision_out[0].shape[0], n_merged)
        self.assertEqual(vision_out[0].shape[1], vc.out_hidden_size)

        atol = 1e-2 if precision == "fp16" else 1e-4
        vis_disc = self.get_numpy_discrepancy(pt_vis_features, vision_out[0].astype(np.float32))
        self.log_results({"step": "vision_encoder", **vis_disc, **log_data})
        np.testing.assert_allclose(pt_vis_features, vision_out[0].astype(np.float32), atol=atol)

        # --- Run audio encoder and compare to PyTorch ---
        # Use n_frames=100 (single chunk, well within n_window * 2 = 200).
        ac = config.audio_config
        n_frames = 100  # single chunk: n_frames <= n_window * 2 = 200
        n_conv_out = (n_frames - 1) // 2 + 1  # after conv2 with stride=2
        n_pooled = (n_conv_out - 2) // 2 + 1  # after AvgPool1d(kernel=2, stride=2)

        torch.manual_seed(1)
        # input_features: [num_mel_bins, n_frames] for audio_tower (single audio, 2D transposed)
        input_features_2d = torch.randn(ac.num_mel_bins, n_frames)

        # PyTorch reference: call audio_tower directly with single-chunk inputs.
        feature_lens = torch.tensor([n_frames], dtype=torch.long)
        aftercnn_lens = torch.tensor([n_conv_out], dtype=torch.long)
        with torch.no_grad():
            pt_aud_out = model.audio_tower(input_features_2d, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens)
        pt_aud_features = pt_aud_out.last_hidden_state.numpy().astype(np.float32)

        audio_sess = self.check_ort(audio_onnx_path)
        audio_out = audio_sess.run(None, {"input_features": input_features_2d.numpy().astype(np.float32)})
        self.assertIsNotNone(audio_out[0])
        self.assertEqual(audio_out[0].shape[0], n_pooled)
        self.assertEqual(audio_out[0].shape[1], ac.output_dim)

        atol_audio = 1e-2 if precision == "fp16" else 1e-4
        aud_disc = self.get_numpy_discrepancy(pt_aud_features, audio_out[0].astype(np.float32))
        self.log_results({"step": "audio_encoder", **aud_disc, **log_data})
        np.testing.assert_allclose(pt_aud_features, audio_out[0].astype(np.float32), atol=atol_audio)

        # --- Run text decoder forward pass ---
        batch_size = 1
        seq_len = 5
        input_ids_t = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_ids_t).numpy().astype(np_dtype)

        text_sess = self.check_ort(text_onnx_path)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}
        num_kv_heads = config.text_config.num_key_value_heads
        head_size_text = config.text_config.hidden_size // config.text_config.num_attention_heads

        position_ids_3d = np.tile(np.arange(seq_len, dtype=np.int64), (3, batch_size, 1))
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids_3d,
        }
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_kv_heads, 0, head_size_text), dtype=np_dtype)
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_kv_heads, 0, head_size_text), dtype=np_dtype)
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        text_out = text_sess.run(None, onnx_feed)
        self.assertIsNotNone(text_out[0])

    @hide_stdout()
    def test_qwen25omni_conditional_generation_fp32_cpu(self):
        self.common_qwen25omni_conditional_generation("fp32", "cpu")

    @hide_stdout()
    def test_qwen25omni_conditional_generation_fp16_cpu(self):
        self.common_qwen25omni_conditional_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_qwen25omni_conditional_generation_fp16_cuda(self):
        self.common_qwen25omni_conditional_generation("fp16", "cuda")

    def common_qwen25omni_conditional_generation_genai(self, precision, provider):
        """Build the Qwen2.5-Omni multimodal pipeline and run it via ORT GenAI.

        Exports ``vision_encoder.onnx``, ``audio_encoder.onnx``,
        ``embedding.onnx``, and ``model.onnx`` then loads them with
        ``onnxruntime_genai``.  A text-only prompt is used so that neither
        the vision nor the audio encoder is invoked (ORT-GenAI's phi3v
        model type cannot pass ``rotary_pos_emb`` to the vision encoder).

        PyTorch parity is not checked: the mRoPE position-id computation
        differs between PyTorch (``get_rope_index``) and ORT-GenAI.  The
        test verifies that ORT-GenAI can load the full four-model pipeline
        (vision + audio + embedding + text) and generate ``max_new_tokens``
        tokens without error.
        """
        import torch
        import onnxruntime_genai as og
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            PreTrainedTokenizerFast,
            Qwen2_5OmniAudioEncoderConfig,
            Qwen2_5OmniThinkerConfig,
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniVisionEncoderConfig,
        )

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        audio_config = Qwen2_5OmniAudioEncoderConfig(
            encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=32, output_dim=256
        )
        vision_config = Qwen2_5OmniVisionEncoderConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256)
        config = Qwen2_5OmniThinkerConfig(audio_config=audio_config, vision_config=vision_config)
        config.text_config.hidden_size = 256
        config.text_config.intermediate_size = 512
        config.text_config.num_hidden_layers = num_hidden_layers
        config.text_config.layer_types = ["full_attention"] * num_hidden_layers
        config.text_config.num_attention_heads = 4
        config.text_config.num_key_value_heads = 2
        config.text_config.vocab_size = 32000
        config.text_config.bos_token_id = 1
        config.text_config.eos_token_id = 2
        config.text_config.rms_norm_eps = 1e-6
        config.text_config.max_position_embeddings = 2048
        config.text_config.hidden_act = "silu"
        config.text_config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [8, 12, 12]}
        config.architectures = ["Qwen2_5OmniForConditionalGeneration"]

        prefix = f"test_qwen25omni_cond_gen_genai_{precision}_{provider}"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)

        torch.manual_seed(42)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
            multimodal=True,
        )

        # --- Text-only prompt: plain token IDs, no image/audio placeholders ---
        # ORT-GenAI's phi3v pipeline cannot pass "rotary_pos_emb" to the
        # vision encoder, so we avoid triggering the vision encoder by
        # using only regular text tokens.
        max_new_tokens = 3
        full_prompt_ids = np.array([10, 20, 100, 200], dtype=np.int64)

        # --- Run ORT GenAI ---
        og_model = og.Model(output_dir)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=len(full_prompt_ids) + max_new_tokens, temperature=1.0, top_k=1)
        generator = og.Generator(og_model, params)
        generator.append_tokens(full_prompt_ids)
        og_generated = []
        while not generator.is_done():
            generator.generate_next_token()
            og_generated.append(int(generator.get_next_tokens()[0]))

        log_data = dict(
            precision=precision,
            model_id=QWEN2_5_OMNI_MODEL_NAME,
            experiment="genai_multimodal_text_generate",
            provider=provider,
            test=prefix,
            input_type="text",
            kind="fast",
        )
        self.log_results({**log_data, "n_generated": len(og_generated)})
        self.assertEqual(len(og_generated), max_new_tokens)

    @hide_stdout()
    @requires_genai()
    def test_qwen25omni_conditional_generation_fp32_cpu_genai(self):
        self.common_qwen25omni_conditional_generation_genai("fp32", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_qwen25omni_conditional_generation_fp16_cpu_genai(self):
        self.common_qwen25omni_conditional_generation_genai("fp16", "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
