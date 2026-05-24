# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

QWEN3_5_MODEL_NAME = "Qwen/Qwen3.5-3B"


def _make_qwen3_5_config(layer_types, num_hidden_layers=None):
    """Return a minimal ``Qwen3_5Config`` suitable for offline unit tests.

    Parameters
    ----------
    layer_types:
        List of layer type strings, e.g. ``["full_attention", "linear_attention"]``.
        The number of layers is inferred from this list unless ``num_hidden_layers``
        is provided.
    num_hidden_layers:
        Explicit override; defaults to ``len(layer_types)``.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)

    # partial_rotary_factor=0.25, head_dim=64 → rdim=16, rdim_half=8.
    # mrope_section=[2, 3, 3]: height positions at stride-1 offsets within rdim_half.
    rope_cfg = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 10000.0, "partial_rotary_factor": 0.25}

    text_config = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
        vocab_size=32000,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    text_config.rope_scaling = rope_cfg
    text_config.rope_parameters = rope_cfg

    config = Qwen3_5Config(text_config=text_config, bos_token_id=1, eos_token_id=2)
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    return config


class TestRandomQwen3_5(ExtTestCase):
    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_and_save_model(self, config, precision, provider):
        """Create a random-weight HF model and build its ONNX export.

        Returns the (model, output_dir) tuple so callers can run inference.
        """
        import torch
        from transformers import AutoModelForImageTextToText

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_{precision}_{provider}_{'_'.join(config.text_config.layer_types)}"
        model_dir_full = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_config(config)
        model.eval()
        model.save_pretrained(model_dir_full)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir_full)

        create_model(
            model_name=QWEN3_5_MODEL_NAME,
            input_path=model_dir_full,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return model, output_dir

    def _run_text_decoder(self, model, output_dir, config, precision, layer_types, cpu=True):
        """Load the ONNX text decoder and run a single prefill step.

        Returns the ONNX output list.
        """
        import torch

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        text_sess = self._check_with_ort(text_onnx_path, cpu=cpu)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        batch_size = 1
        seq_len = 5
        text_cfg = config.text_config

        # Compute inputs_embeds from the saved model's embedding layer.
        torch.manual_seed(0)
        input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy().astype(self.get_input_np_dtype(precision))

        # 3D position_ids [3, batch_size, seq_len] for mRoPE.
        # For a plain text prompt all three rows (temporal / height / width) are identical.
        pos = np.arange(seq_len, dtype=np.int64)
        position_ids_3d = np.stack([pos, pos, pos], axis=0)  # [3, seq_len]
        position_ids_3d = np.stack([position_ids_3d] * batch_size, axis=1)  # [3, B, S]

        # linear_conv_dim = linear_num_key_heads * linear_key_head_dim * 2
        #                   + linear_num_value_heads * linear_value_head_dim
        linear_conv_dim = (
            text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim * 2
            + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
        )
        conv_kernel_minus1 = text_cfg.linear_conv_kernel_dim - 1

        np_dtype = self.get_input_np_dtype(precision)
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids_3d,
        }

        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
                onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
            else:
                # linear_attention: conv_state + recurrent_state
                onnx_feed[f"past_key_values.{i}.conv_state"] = np.zeros((batch_size, linear_conv_dim, conv_kernel_minus1), dtype=np_dtype)
                onnx_feed[f"past_key_values.{i}.recurrent_state"] = np.zeros(
                    (batch_size, text_cfg.linear_num_value_heads, text_cfg.linear_key_head_dim, text_cfg.linear_value_head_dim),
                    dtype=np_dtype,
                )

        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}
        outputs = text_sess.run(None, onnx_feed)
        return outputs

    # ------------------------------------------------------------------ #
    # Tests: full_attention only (standard ORT, no custom ops needed)     #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_fp32_cpu_full_attention(self):
        """Build and run a Qwen3.5 text decoder with only full_attention layers.

        Using all full_attention layers avoids the ``com.microsoft:CausalConvWithState``
        and ``com.microsoft:LinearAttention`` custom ops so that the test can run
        with the standard ``onnxruntime`` Python binding.

        This exercises the mRoPE embedding, the QK-norm in full attention, the
        OffsetRMSNorm (1+weight) layernorm, the 3-D position_ids input, and the
        ``inputs_embeds``-based interface.
        """
        config = _make_qwen3_5_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        outputs = self._run_text_decoder(model, output_dir, config, "fp32", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        # logits: [batch_size, seq_len, vocab_size]
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_fp16_cpu_full_attention(self):
        """fp16 variant of :meth:`test_qwen3_5_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp16", "cpu")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    @requires_cuda()
    def test_qwen3_5_fp16_cuda_full_attention(self):
        """fp16 / CUDA variant of :meth:`test_qwen3_5_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp16", "cuda")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", ["full_attention", "full_attention"], cpu=False)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    # ------------------------------------------------------------------ #
    # Tests: hybrid architecture build verification                       #
    # The linear_attention layers use com.microsoft:CausalConvWithState   #
    # and com.microsoft:LinearAttention custom ops.  When ORT < 1.26,    #
    # both ops are provided as ONNX local-function fallbacks so inference #
    # runs on the standard onnxruntime CPU EP as well.                    #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_fp32_cpu_hybrid_build(self):
        """Verify that ``create_model`` builds a hybrid Qwen3.5 model.

        The hybrid architecture mixes one ``full_attention`` layer and one
        ``linear_attention`` layer.  Both ``com.microsoft:CausalConvWithState``
        and ``com.microsoft:LinearAttention`` custom ops are expected in the
        graph.  When ORT < 1.26, the builder embeds ONNX local-function
        fallbacks for both ops so inference also runs on standard
        ``onnxruntime``.
        """
        import onnx

        config = _make_qwen3_5_config(["full_attention", "linear_attention"])
        model, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        # Check that the ONNX file is a valid protobuf model.
        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        # Confirm that the linear-attention custom ops are present in the graph.
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

        # Run a prefill step to confirm the local-function fallbacks work.
        outputs = self._run_text_decoder(model, output_dir, config, "fp32", config.text_config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_fp16_cpu_hybrid_build(self):
        """fp16 variant of :meth:`test_qwen3_5_fp32_cpu_hybrid_build`."""
        import onnx

        config = _make_qwen3_5_config(["full_attention", "linear_attention"])
        model, output_dir = self._build_and_save_model(config, "fp16", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", config.text_config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_fp16_no_layernorm_fp32_casts(self):
        """Verify that the fp16 build does not wrap LayerNorm IO with fp32 Casts.

        ``Qwen3_5TextModel`` previously forced ``use_fp32`` on every RMSNorm,
        which inserted ~216 Cast nodes (to-fp32 + to-fp16) around the LayerNorm
        ops in a 24-layer build and measurably hurt generation throughput.  See
        microsoft/onnxruntime-genai#2101.  When the runtime is ORT >= 1.26 the
        builder keeps LayerNorm IO in the model's native dtype: no Cast-to-fp32
        should appear on the inputs of the ``SkipSimplifiedLayerNormalization``
        ops, and no Cast-from-fp32 should appear on their outputs.  On older
        ORT releases the fp32 cast wrapping is still emitted (the native fp16
        kernel loses precision across the 36+ Qwen3.5 layers), so this test is
        skipped there.
        """
        import onnx
        import onnxruntime as ort

        ort_ver = tuple(int(x) for x in ort.__version__.split(".")[:2])
        if ort_ver < (1, 26):
            self.skipTest(f"requires onnxruntime >= 1.26, got {ort.__version__}")

        config = _make_qwen3_5_config(["full_attention", "full_attention"])
        _, output_dir = self._build_and_save_model(config, "fp16", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        nodes_by_output = {out: node for node in onnx_model.graph.node for out in node.output}

        FLOAT = onnx.TensorProto.FLOAT
        offending = []
        for node in onnx_model.graph.node:
            if node.op_type != "SkipSimplifiedLayerNormalization":
                continue
            # No upstream Cast-to-fp32 feeding the LayerNorm inputs.
            for inp in node.input:
                producer = nodes_by_output.get(inp)
                if producer is None or producer.op_type != "Cast":
                    continue
                to_attr = next((a.i for a in producer.attribute if a.name == "to"), None)
                if to_attr == FLOAT:
                    offending.append((node.name, "input", producer.name))
            # No downstream Cast consuming the LayerNorm outputs from fp32.
            for out in node.output:
                if not out:
                    continue
                for consumer in onnx_model.graph.node:
                    if consumer.op_type != "Cast" or out not in consumer.input:
                        continue
                    # Cast from fp32 means the LayerNorm output was fp32.
                    val_info = next((v for v in list(onnx_model.graph.value_info) + list(onnx_model.graph.output) if v.name == out), None)
                    if val_info is not None and val_info.type.tensor_type.elem_type == FLOAT:
                        offending.append((node.name, "output", consumer.name))

        self.assertEqual(offending, [], f"Unexpected fp32 Cast nodes around LayerNorm: {offending}")

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_qk_l2norm_uses_lp_normalization(self):
        """Verify the qk_l2norm path emits a single LpNormalization op.

        Ported from microsoft/onnxruntime-genai#2127: the previous
        5-node Square + ReduceSum + Add(eps) + Rsqrt + Mul subgraph used
        per Q/K head per layer is replaced with a single
        ``LpNormalization(p=2, axis=-1)`` op, which is natively supported
        by all current EPs (CPU, CUDA, WebGPU, ...).  This shaves ~5 ops
        per Q/K head per linear-attention layer and drops the +eps fallback
        (q/k come from RMSNorm+Proj so magnitudes far exceed 1e-6).
        """
        import onnx

        config = _make_qwen3_5_config(["full_attention", "linear_attention"])
        _, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        lp_norm_nodes = [n for n in onnx_model.graph.node if n.op_type == "LpNormalization"]

        # Each linear-attention layer L emits two qk_l2norm LpNormalization
        # nodes (one for Q, one for K). The hybrid config has 1 such layer.
        self.assertGreaterEqual(len(lp_norm_nodes), 2, "expected at least 2 LpNormalization nodes for qk_l2norm Q/K")

        # All qk_l2norm LpNormalization nodes must have axis=-1 and p=2.
        for node in lp_norm_nodes:
            if "l2norm" not in node.name:
                continue
            axis = next((a.i for a in node.attribute if a.name == "axis"), None)
            p = next((a.i for a in node.attribute if a.name == "p"), None)
            self.assertEqual(axis, -1, f"{node.name}: axis must be -1")
            self.assertEqual(p, 2, f"{node.name}: p must be 2")

        # The old 5-node subgraph names must no longer appear under any
        # ``*_l2norm`` basename.
        for node in onnx_model.graph.node:
            if "l2norm" not in node.name:
                continue
            for suffix in ("/Square/Mul", "/SumSq/ReduceSum", "/AddEps/Add", "/Rsqrt", "/Normalize/Mul"):
                self.assertFalse(
                    node.name.endswith(suffix),
                    f"qk_l2norm path still emits legacy subgraph node {node.name}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
