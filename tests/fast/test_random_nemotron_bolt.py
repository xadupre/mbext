# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_onnx_light, requires_transformers

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"


@requires_transformers("5.19")
class TestRandomNemotronBolt(ExtTestCase):
    def _export(self, precision="fp32", provider="cpu", **extra_options):
        import torch
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

        from modelbuilder.builder import create_model

        text_config = Gemma4TextConfig(
            vocab_size=193,
            vocab_size_per_layer_input=193,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=64,
            hidden_size_per_layer_input=16,
            num_kv_shared_layers=3,
            layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
            max_position_embeddings=128,
            sliding_window=16,
            final_logit_softcapping=30.0,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
            },
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        config = Gemma4Config(text_config=text_config, vision_config=None, audio_config=None)
        config.architectures = ["Gemma4ForConditionalGeneration"]
        torch.manual_seed(42)
        model = Gemma4ForConditionalGeneration(config).eval().to(torch.bfloat16)
        prefix = f"{self._testMethodName}_{precision}_{provider}"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1).save_pretrained(model_dir)
        model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            use_qdq=precision == "int4",
            **extra_options,
        )
        return model, output_dir

    @staticmethod
    def _quantize_reference(model):
        import torch

        from modelbuilder.helpers.quantization import _default_quantize

        # Compare INT4 execution with real HF arithmetic on the same quantized
        # weights, not with greedy decisions from unquantized random weights.
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            weight = module.weight.detach().float().T.contiguous().numpy()
            packed, scales, _ = _default_quantize(weight, 4, 32, True, True)
            codes = np.stack((packed & 15, packed >> 4), axis=1).reshape(-1)[: weight.size]
            signed = ((codes.astype(np.int16) + 8) % 16 - 8).reshape(weight.shape)
            dequantized = signed * np.repeat(scales, 32, axis=0)[: weight.shape[0]]
            module.weight = torch.nn.Parameter(torch.from_numpy(dequantized.T.copy()), requires_grad=False)

    def _check_prefill_decode(self, precision, provider="cpu"):
        import torch

        from modelbuilder.helpers.rt_helper import _ort_type_to_numpy_dtype

        model, output_dir = self._export(precision, provider)
        if precision == "int4":
            self._quantize_reference(model)
        session = self.check_ort(os.path.join(output_dir, "model.onnx"), provider=provider)
        names = {value.name for value in session.get_inputs()}
        self.assertIn("input_ids", names)
        self.assertNotIn("inputs_embeds", names)
        state = {}
        for value in session.get_inputs():
            if value.name.startswith("past_key_values."):
                state[value.name] = np.zeros(
                    [1 if axis == 0 else dim if isinstance(dim, int) else 0 for axis, dim in enumerate(value.shape)],
                    dtype=_ort_type_to_numpy_dtype(value.type),
                )
        self.assertEqual({value.shape[-1] for value in state.values()}, {32, 64})
        if precision == "int4":
            import onnx_light.onnx as onnx

            graph = onnx.load(os.path.join(output_dir, "model.onnx"), load_external_data=False).graph
            self.assertTrue(any(node.op_type == "DequantizeLinear" for node in graph.node))
        past = None
        position = 0
        for ids in ([3, 7, 11, 5], [13], [17], [19]):
            input_ids = torch.tensor([ids])
            with torch.no_grad():
                reference = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past = reference.past_key_values
            feeds = {
                "input_ids": input_ids.numpy(),
                "attention_mask": np.ones((1, position + len(ids)), dtype=np.int64),
                "position_ids": np.arange(position, position + len(ids), dtype=np.int64)[None],
                **state,
            }
            outputs = dict(
                zip([value.name for value in session.get_outputs()], session.run(None, {k: v for k, v in feeds.items() if k in names}))
            )
            # ORT's optimized QDQ MatMuls can quantize activations as well.
            tolerance = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 3e-2, "int4": 0.1}[precision]
            with self.subTest(position=position):
                self.assertEqualArray(reference.logits.numpy(), outputs["logits"].astype(np.float32), atol=tolerance, rtol=tolerance)
                if precision == "fp32":
                    for name in state:
                        layer_id = int(name.split(".")[1])
                        if layer_id < len(past.layers):
                            cache = past.layers[layer_id]
                            expected = cache.keys if name.endswith(".key") else cache.values
                            self.assertEqualArray(
                                expected.numpy(), outputs[name.replace("past_key_values.", "present.")], atol=1e-4, rtol=1e-4
                            )
            state = {name: outputs[name.replace("past_key_values.", "present.")] for name in state}
            position += len(ids)

    @hide_stdout()
    def test_nemotron_bolt_fp32_cpu(self):
        self._check_prefill_decode("fp32")

    @hide_stdout()
    def test_nemotron_bolt_fp16_cpu(self):
        self._check_prefill_decode("fp16")

    @hide_stdout()
    def test_nemotron_bolt_int4_cpu(self):
        self._check_prefill_decode("int4")

    @requires_cuda()
    @hide_stdout()
    def test_nemotron_bolt_bf16_cuda(self):
        self._check_prefill_decode("bf16", "cuda")

    @requires_cuda()
    @hide_stdout()
    def test_nemotron_bolt_fp16_cuda(self):
        self._check_prefill_decode("fp16", "cuda")

    @requires_cuda()
    @hide_stdout()
    def test_nemotron_bolt_int4_cuda(self):
        self._check_prefill_decode("int4", "cuda")

    @hide_stdout()
    def test_nemotron_bolt_requires_token_embeddings(self):
        with self.assertRaisesRegex(ValueError, "exclude_embeds=false"):
            self._export(exclude_embeds=True)

    @requires_genai()
    @hide_stdout()
    def test_nemotron_bolt_genai_generate(self):
        self._check_generation(self.run_genai_generation_test)

    @requires_genai()
    @hide_stdout()
    def test_nemotron_bolt_int4_genai_generate(self):
        self._check_generation(self.run_genai_generation_test, "int4")

    @requires_onnx_light("0.2.7", cpu_version="0.2.7")
    @hide_stdout()
    def test_nemotron_bolt_onnx_light_generate(self):
        self._check_generation(self.run_onnx_light_generation_test)

    @requires_onnx_light("0.2.7", cpu_version="0.2.7")
    @hide_stdout()
    def test_nemotron_bolt_int4_onnx_light_generate(self):
        self._check_generation(self.run_onnx_light_generation_test, "int4")

    def _check_generation(self, generation_test, precision="fp32"):
        import torch

        model, output_dir = self._export(precision)
        if precision == "int4":
            self._quantize_reference(model)
        prompt = torch.tensor([[3, 7, 11, 5]])
        with torch.no_grad():
            reference = model.generate(prompt, max_new_tokens=5, do_sample=False, eos_token_id=None, pad_token_id=0)[0].tolist()
        generation_test(output_dir, None, model.config.text_config.vocab_size, None, prompt_ids=prompt, pt_tokens=reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
