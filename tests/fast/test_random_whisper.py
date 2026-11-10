# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_onnx_light

WHISPER_MODEL_NAME = "openai/whisper-tiny"


class TestWhisperModel(ExtTestCase):
    @hide_stdout()
    def test_whisper_fp32_cpu_onnx_light_generate(self):
        self._test_native_generation(native_encoder=True)

    @hide_stdout()
    def test_whisper_decoder_fp32_cpu_onnx_light_generate(self):
        """Real HF audio encoder; native cross-cache decoding, not native encoding."""
        self._test_native_generation(native_encoder=False)

    @requires_onnx_light("0.2.7", cpu_version="0.2.7")
    def _test_native_generation(self, *, native_encoder):
        import torch
        import onnx_light.onnx as onnx
        import onnx_light.onnx.helper as oh
        from onnx_light.onnx.reference import ReferenceEvaluator
        from onnx_light_cpu import register_kernels_global
        from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration
        from transformers.generation import GenerationMixin

        from modelbuilder.builder import create_model

        register_kernels_global()
        config = WhisperConfig(
            architectures=["WhisperForConditionalGeneration"],
            vocab_size=128,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            num_mel_bins=8,
            max_source_positions=1500,
            max_target_positions=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            decoder_start_token_id=1,
            suppress_tokens=None,
            begin_suppress_tokens=None,
        )
        torch.manual_seed(42)
        model = WhisperForConditionalGeneration(config).eval()
        basename = f"native_whisper_encoder_{native_encoder}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)
        create_model(
            model_name=WHISPER_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )
        audio = np.random.default_rng(42).standard_normal((1, 8, 3000)).astype(np.float32)
        prompt = np.array([[1, 3, 4]], dtype=np.int64)
        # Generic seq2seq generation avoids Whisper's timestamp/long-form policy,
        # which requires tokenizer assets not present in an offline tiny fixture.
        with torch.no_grad():
            encoder_output = model.model.encoder(torch.from_numpy(audio))
            reference = GenerationMixin.generate(
                model,
                encoder_outputs=encoder_output,
                decoder_input_ids=torch.from_numpy(prompt).long(),
                generation_config=GenerationConfig(
                    decoder_start_token_id=1, bos_token_id=1, pad_token_id=0, eos_token_id=None, max_new_tokens=5, do_sample=False
                ),
            )[0].tolist()
            cross_cache = {}
            if not native_encoder:
                for role, projection in (
                    ("key", model.model.decoder.layers[0].encoder_attn.k_proj),
                    ("value", model.model.decoder.layers[0].encoder_attn.v_proj),
                ):
                    cross_cache[f"present_{role}_cross_0"] = (
                        projection(encoder_output.last_hidden_state).reshape(1, 1500, 2, 16).transpose(1, 2).contiguous().numpy()
                    )
        if native_encoder:
            encoder = onnx.load(os.path.join(output_dir, "encoder.onnx"), load_external_data=True)
            encoder_values = ReferenceEvaluator(encoder).run(None, {"audio_features": audio})
            cross_cache = dict(zip((value.name for value in encoder.graph.output), encoder_values))
        decoder = onnx.load(os.path.join(output_dir, "decoder.onnx"), load_external_data=True)
        # Native generation owns INT64 token IDs; preserve Whisper's INT32
        # decoder contract with an explicit native Cast frontend.
        for value in decoder.graph.input:
            if value.name == "input_ids":
                value.name = "decoder_input_ids"
        for node in decoder.graph.node:
            if "input_ids" in node.input:
                inputs = ["decoder_input_ids" if name == "input_ids" else name for name in node.input]
                node.input.clear()
                node.input.extend(inputs)
        token_cast = oh.make_model(
            oh.make_graph(
                [oh.make_node("Cast", ["input_ids"], ["decoder_input_ids"], to=onnx.TensorProto.INT32)],
                "whisper_token_cast",
                [oh.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [None, None])],
                [oh.make_tensor_value_info("decoder_input_ids", onnx.TensorProto.INT32, [None, None])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
        )
        feeds = {"input_ids": prompt}
        for role in ("key", "value"):
            input_name = f"past_{role}_self_0"
            binding = decoder.graph.persistent_bindings.add()
            binding.input_name = input_name
            binding.output_name = f"present_{role}_self_0"
            feeds[input_name] = np.zeros((1, 2, 0, 16), dtype=np.float32)
            feeds[f"past_{role}_cross_0"] = cross_cache[f"present_{role}_cross_0"]
        tokens = ReferenceEvaluator(decoder).generate(feeds, max_new_tokens=5, temperature=0.0, embedding=ReferenceEvaluator(token_cast))
        self.assertEqual(reference, tokens[0].tolist())

    def common_fast_whisper_random_weights(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast, WhisperConfig, WhisperForConditionalGeneration

        from modelbuilder.builder import create_model

        # Tiny Whisper config: 1 encoder layer, 1 decoder layer, small d_model.
        # num_mel_bins and max_source_positions are kept at standard values
        # because the preprocessing Conv layers expect exactly 3000 input frames
        # and max_source_positions = 3000 // 2 = 1500.
        num_encoder_layers = 1
        num_decoder_layers = 1
        config = WhisperConfig(
            architectures=["WhisperForConditionalGeneration"],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            d_model=128,
            encoder_layers=num_encoder_layers,
            encoder_attention_heads=4,
            decoder_layers=num_decoder_layers,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
            num_mel_bins=80,
            max_source_positions=1500,
            max_target_positions=32,
            vocab_size=1000,
        )

        basename = f"test_whisper_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = WhisperForConditionalGeneration(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=WHISPER_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=WHISPER_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="audio",
            kind="fast",
        )

        encoder_path = os.path.join(output_dir, "encoder.onnx")
        decoder_path = os.path.join(output_dir, "decoder.onnx")
        self.assertExists(encoder_path)
        self.assertExists(decoder_path)

        enc_sess = self._check_with_ort(encoder_path, cpu=provider == "cpu")
        dec_sess = self._check_with_ort(decoder_path, cpu=provider == "cpu")

        np_dtype = self.get_input_np_dtype(precision)
        batch_size = 1
        num_attn_heads = config.decoder_attention_heads
        head_size = config.d_model // config.decoder_attention_heads
        seq_len = 3

        enc_input_names = {i.name for i in enc_sess.get_inputs()}
        enc_output_names = [i.name for i in enc_sess.get_outputs()]
        dec_input_names = {i.name for i in dec_sess.get_inputs()}
        dec_output_names = [i.name for i in dec_sess.get_outputs()]

        # ------------------------------------------------------------------
        # Step 1: Run encoder with random audio features
        # ------------------------------------------------------------------
        rng = np.random.default_rng(42)
        audio_np = rng.standard_normal((batch_size, config.num_mel_bins, 3000)).astype(np_dtype)

        enc_feed = {"audio_features": audio_np}
        enc_feed = {k: v for k, v in enc_feed.items() if k in enc_input_names}
        enc_outputs = enc_sess.run(None, enc_feed)
        enc_results = dict(zip(enc_output_names, enc_outputs))

        # Compare encoder hidden_states with PyTorch
        with torch.no_grad():
            pt_audio = torch.from_numpy(audio_np.astype(np.float32)).to(provider)
            pt_enc_out = model.model.encoder(pt_audio)
        pt_hidden = pt_enc_out.last_hidden_state.detach().cpu().numpy()

        enc_disc = self.get_numpy_discrepancy(pt_hidden.astype(np_dtype), enc_results["hidden_states"])
        self.log_results({"step": "encoder", **enc_disc, **log_data})

        atol_enc = {"fp32": 1e-4, "fp16": 5e-2, "int4": 0.5}
        np.testing.assert_allclose(pt_hidden.astype(np_dtype), enc_results["hidden_states"], atol=atol_enc[precision], rtol=1)

        # ------------------------------------------------------------------
        # Step 2: Run decoder prefill with cross-attention KV from encoder
        # ------------------------------------------------------------------
        input_ids = np.array([[1, 2, 3]], dtype=np.int32)

        dec_feed = {"input_ids": input_ids}
        # Empty self-attention KV caches (no prior decode steps)
        for i in range(num_decoder_layers):
            dec_feed[f"past_key_self_{i}"] = np.zeros((batch_size, num_attn_heads, 0, head_size), dtype=np_dtype)
            dec_feed[f"past_value_self_{i}"] = np.zeros((batch_size, num_attn_heads, 0, head_size), dtype=np_dtype)
        # Cross-attention KV caches from the encoder pass
        for i in range(num_encoder_layers):
            dec_feed[f"past_key_cross_{i}"] = enc_results[f"present_key_cross_{i}"]
            dec_feed[f"past_value_cross_{i}"] = enc_results[f"present_value_cross_{i}"]

        dec_feed = {k: v for k, v in dec_feed.items() if k in dec_input_names}
        dec_outputs = dec_sess.run(None, dec_feed)
        dec_results = dict(zip(dec_output_names, dec_outputs))

        # Compare decoder logits with PyTorch
        with torch.no_grad():
            pt_input_ids = torch.from_numpy(input_ids.astype(np.int64)).to(provider)
            pt_dec_out = model.model.decoder(input_ids=pt_input_ids, encoder_hidden_states=pt_enc_out.last_hidden_state)
        pt_logits = model.proj_out(pt_dec_out.last_hidden_state).detach().cpu().numpy()

        dec_disc = self.get_numpy_discrepancy(pt_logits.astype(np_dtype), dec_results["logits"])
        self.log_results({"step": "decoder", **dec_disc, **log_data})

        atol_dec = {"fp32": 1e-3, "fp16": 5e-2, "int4": 0.5}
        np.testing.assert_allclose(pt_logits.astype(np_dtype), dec_results["logits"], atol=atol_dec[precision], rtol=1e-3)

        # Verify KV cache shapes from decoder
        for i in range(num_decoder_layers):
            expected_kv_shape = (batch_size, num_attn_heads, seq_len, head_size)
            self.assertEqual(dec_results[f"present_key_self_{i}"].shape, expected_kv_shape)
            self.assertEqual(dec_results[f"present_value_self_{i}"].shape, expected_kv_shape)

    @hide_stdout()
    @unittest.skip("issue with Attention, onnxruntime is crashing")
    def test_fast_whisper_random_weights_fp32_cpu(self):
        self.common_fast_whisper_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_whisper_random_weights_fp16_cpu(self):
        self.common_fast_whisper_random_weights("fp16", "cpu")

    @hide_stdout()
    @unittest.skip("AttributeError: 'Linear' object has no attribute 'bits'. Did you mean: 'bias'?")
    def test_fast_whisper_random_weights_int4_cpu(self):
        self.common_fast_whisper_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_whisper_random_weights_fp32_cuda(self):
        self.common_fast_whisper_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_whisper_random_weights_fp16_cuda(self):
        self.common_fast_whisper_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
