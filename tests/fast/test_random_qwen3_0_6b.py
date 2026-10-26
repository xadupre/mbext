# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

QWEN3_MODEL_NAME = "Qwen/Qwen3-0.6B"


class TestRandomQwen3(ExtTestCase):
    @hide_stdout()
    def test_parallel_qwen3_multilayer(self):
        import onnx_light.onnx as onnx
        import onnxruntime as ort
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=64,
            intermediate_size=160,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            vocab_size=97,
            max_position_embeddings=128,
            attn_implementation="eager",
        )
        for precision in ("fp32", "fp16", "int4"):
            with self.subTest(precision=precision):
                model = Qwen3ForCausalLM(config).eval().to(torch.bfloat16)
                self.run_random_weights_test(
                    model=model,
                    tokenizer=self.make_word_level_tokenizer(),
                    model_name=QWEN3_MODEL_NAME,
                    basename=f"test_parallel_qwen3_multilayer_{precision}",
                    precision=precision,
                    provider="cpu",
                    num_hidden_layers=config.num_hidden_layers,
                    num_key_value_heads=config.num_key_value_heads,
                    head_size=config.head_dim,
                    vocab_size=config.vocab_size,
                    reference_dtype=torch.float32,
                )
                if precision == "fp16" and tuple(int(x) for x in ort.__version__.split(".")[:2]) >= (1, 26):
                    output_dir, _ = self.get_dirs(f"test_parallel_qwen3_multilayer_{precision}")
                    exported = onnx.load(os.path.join(output_dir, "model.onnx"))
                    nodes = exported.graph.node
                    producers = {out: node for node in nodes for out in node.output if out}
                    initializers = {init.name: init for init in exported.graph.initializer}
                    norms = [node for node in nodes if node.op_type in ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization")]
                    self.assertTrue(any(node.op_type == "SimplifiedLayerNormalization" for node in norms))
                    self.assertTrue(any(node.op_type == "SkipSimplifiedLayerNormalization" for node in norms))
                    for node in norms:
                        with self.subTest(node=node.name):
                            weight_index = 2 if node.op_type.startswith("Skip") else 1
                            self.assertEqual(initializers[node.input[weight_index]].data_type, onnx.TensorProto.FLOAT16)
                            for inp in list(node.input)[:weight_index]:
                                producer = producers.get(inp)
                                self.assertTrue(producer is None or producer.op_type != "Cast", node.name)
                            for consumer in nodes:
                                if any(out and out in consumer.input for out in node.output):
                                    self.assertNotEqual(consumer.op_type, "Cast", node.name)
                            if not node.op_type.startswith("Skip"):
                                self.assertEqual(next(a.i for a in node.attribute if a.name == "stash_type"), onnx.TensorProto.FLOAT)

    def common_fast_qwen3_random_weights(self, precision, provider, model_dtype=None, reference_dtype=None):
        from transformers import AutoModelForCausalLM, Qwen3Config

        num_hidden_layers = 1

        # Minimal Qwen3 config matching the Qwen3ForCausalLM architecture but
        # with small dimensions to keep the test fast and completely offline.
        # head_dim=64 matches hidden_size // num_attention_heads = 512 // 8.
        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=32000,
            use_sliding_window=False,
        )

        model = AutoModelForCausalLM.from_config(config)
        if model_dtype is not None:
            model.to(model_dtype)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN3_MODEL_NAME,
            basename=f"test_discrepancies_qwen3_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            reference_dtype=reference_dtype,
        )

    def common_qwen3_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, Qwen3Config

        num_hidden_layers = 1

        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=32000,
            use_sliding_window=False,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN3_MODEL_NAME,
            basename=f"test_generation_qwen3_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_qwen3_fp32_cpu_greedy_generation(self):
        self.common_qwen3_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_qwen3_fp16_cpu_greedy_generation(self):
        self.common_qwen3_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_qwen3_fp32_cuda_greedy_generation(self):
        self.common_qwen3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_qwen3_fp16_cuda_greedy_generation(self):
        self.common_qwen3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_qwen3_bf16_cuda_greedy_generation(self):
        self.common_qwen3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_fp32_cpu(self):
        self.common_fast_qwen3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_fp16_cpu(self):
        self.common_fast_qwen3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_bf16_weights_fp16_cpu(self):
        import torch

        self.common_fast_qwen3_random_weights("fp16", "cpu", model_dtype=torch.bfloat16, reference_dtype=torch.float32)

    @hide_stdout()
    def test_fast_discrepancy_qwen3_int4_cpu(self):
        self.common_fast_qwen3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_int4_cuda(self):
        self.common_fast_qwen3_random_weights("int4", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_fp16_cuda(self):
        self.common_fast_qwen3_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_qwen3_fp32_cpu_genai_generate(self):
        return self._test_qwen3_fp32_cpu_generate(self.run_genai_generation_test, "test_qwen3_fp32_cpu_genai_generate")

    @hide_stdout()
    def test_qwen3_fp32_cpu_onnx_light_generate(self):
        return self._test_qwen3_fp32_cpu_generate(self.run_onnx_light_generation_test, "test_qwen3_fp32_cpu_onnx_light_generate")

    def _test_qwen3_fp32_cpu_generate(self, generation_test, test_name):
        import torch
        from transformers import AutoModelForCausalLM, Qwen3Config

        from modelbuilder.builder import create_model

        prefix = test_name
        num_hidden_layers = 1
        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=32000,
            use_sliding_window=False,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=QWEN3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
