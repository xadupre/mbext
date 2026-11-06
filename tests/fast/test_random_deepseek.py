# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Fast random-weight tests for DeepSeek-V3 / DeepSeek-V4-Flash.

These tests use a tiny synthetic config to validate that the builder can
export an ONNX model and that PyTorch vs ONNX logit discrepancies are within
acceptable tolerances.

Architecture strings dispatched to the same builder (DeepSeekV3Model):
  - ``DeepseekV3ForCausalLM`` (DeepSeek-V3 / deepseek_v3 transformers model type)
  - ``DeepseekV4ForCausalLM`` (DeepSeek-V4-Flash — same MLA+MoE structure as V3)

Note: ``DeepseekV4ForCausalLM`` is not yet registered in the transformers
model registry, so the synthetic config is created with
``deepseek_v3`` as the ``model_type`` and the architecture string is
patched afterwards.  This exercises the builder dispatch path for V4 without
requiring a separate registered config class.
"""

import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_onnx_light, requires_transformers

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"


def _make_tiny_deepseek_config(**overrides):
    """Return a tiny DeepseekV3Config suitable for random-weight tests."""
    from transformers import AutoConfig

    cfg_kwargs = dict(
        architectures=["DeepseekV3ForCausalLM"],
        hidden_size=64,
        intermediate_size=64,  # dense MLP (first_k_dense_replace layers)
        moe_intermediate_size=16,  # per-expert intermediate size
        num_hidden_layers=3,  # 1 dense + 2 MoE
        num_attention_heads=4,
        num_key_value_heads=4,
        # MLA dimensions
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        # MoE
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        # Other
        hidden_act="silu",
        rms_norm_eps=1e-5,
        vocab_size=256,
        max_position_embeddings=128,
        rope_interleave=True,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg_kwargs.update(overrides)
    # Always use the "deepseek_v3" model_type regardless of the architecture
    # string.  DeepseekV4ForCausalLM is not yet registered in the transformers
    # model registry, so its config must be built via deepseek_v3 and then the
    # architectures field overridden (see test_fast_discrepancy_deepseek_v4_*).
    return AutoConfig.for_model("deepseek_v3", **cfg_kwargs)


@requires_transformers("5")
class TestDeepSeekV3(ExtTestCase):
    @hide_stdout()
    def test_deepseek_v3_fp32_cpu_onnx_light_generate(self):
        self._test_native_generation("DeepseekV3ForCausalLM")

    @hide_stdout()
    def test_deepseek_v4_fp32_cpu_onnx_light_generate(self):
        self._test_native_generation("DeepseekV4ForCausalLM")

    def _test_native_generation(self, architecture):
        import torch
        from transformers import AutoModelForCausalLM

        torch.manual_seed(42)
        config = _make_tiny_deepseek_config(architectures=[architecture])
        model = AutoModelForCausalLM.from_config(config).eval()
        self.run_onnx_light_model_generation_test(model, MODEL_NAME, f"native_{architecture}")

    @hide_stdout()
    @requires_onnx_light("0.2.7", cpu_version="0.2.7")
    def test_grouped_sigmoid_routing_and_packed_experts(self):
        import numpy as np
        import torch
        from onnx_light.onnx import TensorProto
        from onnx_light.onnx.reference import ReferenceEvaluator
        from onnx_light_cpu import register_kernels_global
        from transformers import AutoModelForCausalLM

        from modelbuilder.builders.deepseek import DeepSeekV3Model

        register_kernels_global()
        _, cache = self.get_dirs("deepseek_routing")
        cases = [(normalize, selected_groups, 8, 2, 2, 5) for normalize in (False, True) for selected_groups in (1, 2)]
        cases.append((True, 2, 256, 8, 8, 64))
        for normalize, selected_groups, num_experts, top_k, groups, sequence in cases:
            with self.subTest(normalize=normalize, selected_groups=selected_groups, num_experts=num_experts):
                torch.manual_seed(42)
                config = _make_tiny_deepseek_config(
                    num_hidden_layers=1,
                    first_k_dense_replace=0,
                    n_routed_experts=num_experts,
                    norm_topk_prob=normalize,
                    topk_group=selected_groups,
                    routed_scaling_factor=2.5,
                    n_group=groups,
                    num_experts_per_tok=top_k,
                )
                mlp = AutoModelForCausalLM.from_config(config).eval().model.layers[0].mlp
                with torch.no_grad():
                    bias = (
                        torch.tensor([0.0, 0.8, -0.3, -0.1, 0.2, -0.4, 0.3, -0.7])
                        if num_experts == 8
                        else torch.linspace(-0.8, 0.8, num_experts)
                    )
                    mlp.gate.e_score_correction_bias.copy_(bias)
                    hidden = torch.randn(2, sequence, config.hidden_size)
                    _, expected_weights, expected_indices = mlp.gate(hidden)
                    expected = mlp(hidden).numpy()
                builder = DeepSeekV3Model(config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", cache, {})
                builder.make_graph_input("hidden", TensorProto.FLOAT, [None, None, config.hidden_size])
                builder.make_moe_layer(0, mlp, "hidden")
                weights = "/model/layers.0/moe/router/scale/output_0"
                indices = "/model/layers.0/moe/router/top_indices"
                builder.make_graph_output(weights, TensorProto.FLOAT, [None, None, top_k])
                builder.make_graph_output(indices, TensorProto.INT64, [None, None, top_k])
                builder.make_graph_output(builder.layernorm_attrs["skip_input"], TensorProto.FLOAT, [None, None, config.hidden_size])
                graph = builder.to_model_proto()
                self.assertEqual(sum(node.op_type == "ScatterND" for node in graph.graph.node), 1)
                initializers = {value.name: value for value in graph.graph.initializer}
                projections = [node for node in graph.graph.node if node.op_type == "MatMul" and "/moe/experts." in node.name]
                self.assertEqual(len(projections), 3 * num_experts)
                self.assertTrue(all(len(initializers[node.input[1]].dims) == 2 for node in projections))
                actual_weights, actual_indices, actual = ReferenceEvaluator(graph).run(None, {"hidden": hidden.numpy()})
                actual_indices = actual_indices.reshape(-1, top_k)
                actual_weights = actual_weights.reshape(-1, top_k)
                order = expected_indices.argsort(-1)
                np.testing.assert_array_equal(np.sort(actual_indices, axis=-1), expected_indices.gather(1, order).numpy())
                np.testing.assert_allclose(
                    np.take_along_axis(actual_weights, actual_indices.argsort(-1), axis=-1),
                    expected_weights.gather(1, order).numpy(),
                    atol=1e-6,
                    rtol=1e-6,
                )
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)

    @hide_stdout()
    @requires_onnx_light("0.2.7", cpu_version="0.2.7")
    def test_selected_experts_int4_quantization(self):
        import os

        import numpy as np
        import torch
        from onnx_light.onnx import TensorProto, load
        from onnx_light.onnx.reference import ReferenceEvaluator
        from onnx_light_cpu import register_kernels_global
        from transformers import AutoModelForCausalLM

        from modelbuilder.builders.deepseek import DeepSeekV3Model

        register_kernels_global()
        for dtype, torch_dtype in ((TensorProto.FLOAT, torch.float32), (TensorProto.FLOAT16, torch.float16)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                config = _make_tiny_deepseek_config(num_hidden_layers=1, first_k_dense_replace=0)
                mlp = AutoModelForCausalLM.from_config(config).to(torch_dtype).eval().model.layers[0].mlp
                hidden = (torch.randn(2, 5, config.hidden_size) * 3).to(torch_dtype).numpy()
                output_dir, cache = self.get_dirs(f"deepseek_experts_int4_{dtype}")
                builder = DeepSeekV3Model(config, dtype, TensorProto.INT4, "cpu", cache, {"int4_accuracy_level": 0})
                builder.make_graph_input("hidden", dtype, [None, None, config.hidden_size])
                builder.make_moe_layer(0, mlp, "hidden")
                builder.make_graph_output(builder.layernorm_attrs["skip_input"], dtype, [None, None, config.hidden_size])
                builder.save_model(output_dir)
                path = os.path.join(output_dir, "model.onnx")
                graph = load(path, load_external_data=True)
                projections = [
                    node for node in graph.graph.node if "/moe/experts." in node.name and node.op_type in {"MatMul", "MatMulNBits"}
                ]
                self.assertEqual(len(projections), 3 * config.num_local_experts)
                self.assertTrue(all(node.op_type == "MatMulNBits" for node in projections))
                router = next(node for node in graph.graph.node if node.name == "/model/layers.0/moe/router/MatMul")
                self.assertEqual(router.op_type, "MatMul")
                expected = self.check_ort(path).run(None, {"hidden": hidden})[0]
                actual = ReferenceEvaluator(graph).run(None, {"hidden": hidden})[0]
                np.testing.assert_allclose(actual, expected, atol=2e-5 if dtype == TensorProto.FLOAT16 else 1e-6, rtol=2e-3)

    def common_fast_deepseek_random_weights(self, precision, provider, arch="DeepseekV3ForCausalLM"):
        from transformers import AutoModelForCausalLM

        num_hidden_layers = 3
        config = _make_tiny_deepseek_config(num_hidden_layers=num_hidden_layers, architectures=[arch])
        # qk_head_dim = qk_nope + qk_rope = 16; used as head_size for KV cache
        kv_head_size = config.qk_head_dim  # = 16

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_deepseek_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=kv_head_size,
            vocab_size=config.vocab_size,
            atol={"fp32": 1e-4, "fp16": 2e-3, "bf16": 1e-2, "int4": 0.5},
            rtol={"fp32": 1e-3, "fp16": 1e-2, "bf16": 1e-2, "int4": 10000},
        )

    @hide_stdout()
    def test_fast_discrepancy_deepseek_fp32_cpu(self):
        self.common_fast_deepseek_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_deepseek_fp16_cpu(self):
        self.common_fast_deepseek_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_deepseek_fp16_cuda(self):
        self.common_fast_deepseek_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_deepseek_bf16_cuda(self):
        self.common_fast_deepseek_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_deepseek_v4_dispatch_fp32_cpu(self):
        # Verify that DeepseekV4ForCausalLM dispatches to the same builder.
        # DeepseekV4ForCausalLM is not yet in the transformers registry, so we
        # create the config via deepseek_v3 and patch the architecture string.
        self.common_fast_deepseek_random_weights("fp32", "cpu", arch="DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
