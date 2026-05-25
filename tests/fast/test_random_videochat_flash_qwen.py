# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

VIDEOCHAT_FLASH_QWEN_MODEL_NAME = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B"


def _force_architectures(model_dir, arch_name):
    """Rewrite ``config.json`` so ``architectures`` matches *arch_name*.

    ``model.save_pretrained`` writes the HF class name (``Qwen2ForCausalLM``)
    into ``architectures``.  The model builder dispatches on this value, so we
    rewrite it to the upstream VLM architecture name to exercise the new
    ``VideoChatFlashQwenModel`` code path.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["architectures"] = [arch_name]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


class TestRandomVideoChatFlashQwen(ExtTestCase):
    """Fast offline tests for the ``VideoChatFlashQwenForCausalLM`` text decoder.

    VideoChat-Flash-Qwen uses a standard Qwen2.5 language backbone (flat
    config, standard weight keys, 2-D RoPE, GQA).  The builder exports only
    the text component of the VLM (with ``exclude_embeds=True``), so the
    resulting ONNX model takes ``inputs_embeds`` rather than ``input_ids``.
    The tests verify that the exported model produces logits numerically
    close to those of the original ``Qwen2ForCausalLM`` HF model.
    """

    @staticmethod
    def _make_config():
        """Return a minimal ``Qwen2Config`` for offline testing."""
        from transformers import Qwen2Config

        return Qwen2Config(
            architectures=["VideoChatFlashQwenForCausalLM"],
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=1,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            vocab_size=32000,
            use_sliding_window=False,
            tie_word_embeddings=False,
        )

    def common_fast_videochat_flash_qwen_random_weights(self, precision, provider):
        from transformers import Qwen2ForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_config()
        num_hidden_layers = config.num_hidden_layers

        model = Qwen2ForCausalLM(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        basename = f"test_discrepancies_videochat_flash_qwen_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        # ``Qwen2ForCausalLM.save_pretrained`` overwrites ``architectures`` with
        # the HF class name; restore the VLM architecture so create_model
        # dispatches to ``VideoChatFlashQwenModel`` (which sets
        # ``exclude_embeds=True``).
        _force_architectures(model_dir, "VideoChatFlashQwenForCausalLM")

        create_model(
            model_name=VIDEOCHAT_FLASH_QWEN_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=VIDEOCHAT_FLASH_QWEN_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        # Confirm the exported model exposes ``inputs_embeds`` (not ``input_ids``).
        sess = self.check_ort(onnx_path, provider=provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]
        self.assertIn("inputs_embeds", onnx_input_names)
        self.assertNotIn("input_ids", onnx_input_names)

        head_size = config.hidden_size // config.num_attention_heads
        self.run_prefill_and_decode_check(
            model=model,
            sess=sess,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.vocab_size,
            precision=precision,
            provider=provider,
            log_data=log_data,
            embed_fn=model.model.embed_tokens,
        )

    @hide_stdout()
    def test_fast_discrepancy_videochat_flash_qwen_fp32_cpu(self):
        self.common_fast_videochat_flash_qwen_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_videochat_flash_qwen_fp16_cpu(self):
        self.common_fast_videochat_flash_qwen_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_videochat_flash_qwen_int4_cpu(self):
        self.common_fast_videochat_flash_qwen_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_videochat_flash_qwen_fp16_cuda(self):
        self.common_fast_videochat_flash_qwen_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
