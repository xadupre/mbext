# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

MODEL_IDS = (
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
)


class TestRandomNemotron3(ExtTestCase):
    def _make_model(self, variant):
        import torch
        from transformers import NemotronHConfig, NemotronHForCausalLM

        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            vocab_size=193,
            hidden_size=64,
            intermediate_size=128,
            layers_block_type=["linear_attention", "mlp" if variant == 0 else "moe", "full_attention", "linear_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            mamba_num_heads=4,
            mamba_head_dim=16,
            ssm_state_size=8,
            n_groups=2,
            conv_kernel=4,
            chunk_size=8,
            use_mamba_kernels=False,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=64,
            moe_latent_size=32 if variant == 2 else None,
            routed_scaling_factor=2.5,
            n_group=2,
            topk_group=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        torch.manual_seed(42)
        model = NemotronHForCausalLM(config).eval()
        with torch.no_grad():
            for layer in model.model.layers:
                if layer.block_type == "moe":
                    layer.mixer.gate.e_score_correction_bias.copy_(torch.tensor([0.3, -0.2, 0.1, 0.4]))
        return model

    def _write_modelopt_checkpoint(self, model, model_dir, variant):
        import torch
        from safetensors.torch import save_file

        state = {}
        reference = {}
        linear_names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

        def add_weight(name, weight, fp8=False, quantized=True):
            source = name.replace("model.", "backbone.", 1) if name.startswith("model.") else name
            base = source.removesuffix(".weight")
            if not quantized:
                state[source] = weight.contiguous()
                return weight
            if fp8:
                scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 448
                packed = (weight / scale).to(torch.float8_e4m3fn)
                state[source] = packed
                state[base + ".weight_scale"] = scale
                return packed.float() * scale
            levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            global_scale = torch.tensor([0.75 + len(state) % 4 * 0.25])
            blocks = weight.reshape(weight.shape[0], -1, 16)
            scale = (blocks.abs().amax(dim=-1) / (6 * global_scale)).clamp_min(1e-4)
            scale = scale.to(torch.float8_e4m3fn)
            normalized = blocks / (scale.float().unsqueeze(-1) * global_scale)
            indices = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(-1)
            codes = (indices | ((normalized < 0).long() << 3)).reshape(weight.shape).to(torch.uint8)
            state[source] = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous()
            state[base + ".weight_scale"] = scale.view(torch.uint8) if variant == 1 else scale
            state[base + ".weight_scale_2"] = global_scale
            values = levels[indices] * torch.where(normalized < 0, -1.0, 1.0)
            return (values * scale.float().unsqueeze(-1) * global_scale).reshape(weight.shape)

        for name, weight in model.state_dict().items():
            if ".experts." in name and weight.ndim == 3:
                prefix, projection = name.rsplit(".", 1)
                reference[name] = torch.stack([add_weight(f"{prefix}.{i}.{projection}.weight", value) for i, value in enumerate(weight)])
            else:
                base = name.removesuffix(".weight")
                quantized = (
                    name.endswith(".weight")
                    and base in linear_names
                    and ".gate" not in name
                    and ".mixer.q_proj" not in name
                    and ".mixer.k_proj" not in name
                    and ".mixer.v_proj" not in name
                    and ".mixer.o_proj" not in name
                    and (name != "lm_head.weight" or variant == 2)
                )
                reference[name] = add_weight(
                    name, weight, fp8=variant == 2 and (".in_proj." in name or ".out_proj." in name), quantized=quantized
                )
        if variant == 2:
            prefix = "backbone.layers.1.mixer.experts"
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                name = f"{prefix}.up_proj" if suffix == "weight" else f"{prefix}.up_proj.{suffix}"
                state[name] = torch.stack([state.pop(f"{prefix}.{i}.up_proj.{suffix}") for i in range(model.config.n_routed_experts)])
            prefix = "backbone.layers.1.mixer.shared_experts.up_proj"
            state[prefix + ".weight_packed"] = state.pop(prefix + ".weight")
            state[prefix + ".weight_global_scale"] = state.pop(prefix + ".weight_scale_2").reciprocal()
        model.load_state_dict(reference)
        if variant == 2:
            model.config.quantization_config = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}
        else:
            with open(os.path.join(model_dir, "hf_quant_config.json"), "w") as stream:
                json.dump({"quantization": {"quant_algo": "NVFP4", "group_size": 16}}, stream)
        model.config.save_pretrained(model_dir)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as stream:
            config = json.load(stream)
        config.pop("layers_block_type")
        config["hybrid_override_pattern"] = "ME*M"
        with open(config_path, "w") as stream:
            json.dump(config, stream)
        # Exercise the sharded layout used by the real 30B checkpoints.
        names = list(state)
        weight_map = {}
        for shard, selected in enumerate((names[::2], names[1::2])):
            filename = f"model-{shard + 1:05d}-of-00002.safetensors"
            save_file({name: state[name] for name in selected}, os.path.join(model_dir, filename))
            weight_map.update({name: filename for name in selected})
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as stream:
            json.dump({"metadata": {}, "weight_map": weight_map}, stream)

    def _export(self, variant, precision="fp32", provider="cpu", modelopt=False):
        from modelbuilder.builder import create_model

        model = self._make_model(variant)
        prefix = f"test_random_nemotron3_{variant}_{precision}_{provider}_{modelopt}"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)
        if modelopt:
            self._write_modelopt_checkpoint(model, model_dir, variant)
        else:
            if variant == 0:
                import torch

                model.to(torch.bfloat16)
            model.save_pretrained(model_dir)
            model.float()
        self.make_word_level_tokenizer().save_pretrained(model_dir)
        create_model(
            model_name=MODEL_IDS[variant],
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return model, output_dir

    def _check_prefill_decode(self, variant, precision="fp32", provider="cpu", modelopt=False):
        import torch

        model, output_dir = self._export(variant, precision, provider, modelopt=modelopt)
        sess = self.check_ort(os.path.join(output_dir, "model.onnx"), provider=provider)
        dtype = np.float16 if precision == "fp16" else np.float32
        batch = 1
        state = {}
        for value in sess.get_inputs():
            if value.name.startswith("past_key_values."):
                shape = [batch if i == 0 else d if isinstance(d, int) else 0 for i, d in enumerate(value.shape)]
                state[value.name] = np.zeros(shape, dtype=np.float32 if value.type == "tensor(float)" else dtype)
        rng = np.random.default_rng(42)
        past = None
        total_length = 0
        for length in (5, 1, 3):
            ids = rng.integers(3, model.config.vocab_size, (batch, length))
            total_length += length
            mask = np.ones((batch, total_length), dtype=np.int64)
            feed = {
                "input_ids": ids,
                "attention_mask": mask,
                "position_ids": np.broadcast_to(np.arange(total_length - length, total_length), ids.shape).copy(),
                **state,
            }
            feed = {v.name: feed[v.name] for v in sess.get_inputs()}
            outputs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feed)))
            with torch.no_grad():
                expected = model(torch.from_numpy(ids), attention_mask=torch.from_numpy(mask), past_key_values=past, use_cache=True)
            past = expected.past_key_values
            self.assertEqual(expected.logits.shape, outputs["logits"].shape)
            self.assertEqualArray(
                expected.logits.float().numpy(),
                outputs["logits"].astype(np.float32),
                atol=3e-3 if precision == "fp16" else 2e-5,
                rtol=3e-3 if precision == "fp16" else 2e-4,
            )
            for name in state:
                state[name] = outputs[name.replace("past_key_values.", "present.")]

    @hide_stdout()
    def test_prefill_decode_fp32_cpu(self):
        for variant in range(3):
            with self.subTest(model=MODEL_IDS[variant]):
                self._check_prefill_decode(variant)

    @hide_stdout()
    def test_prefill_decode_fp16_cpu(self):
        for variant in range(3):
            with self.subTest(model=MODEL_IDS[variant]):
                self._check_prefill_decode(variant, "fp16")

    @hide_stdout()
    def test_modelopt_prefill_decode(self):
        for variant in (1, 2):
            for precision in ("fp32", "fp16"):
                with self.subTest(model=MODEL_IDS[variant], precision=precision):
                    self._check_prefill_decode(variant, precision, modelopt=True)

    @hide_stdout()
    def test_modelopt_invalid_scales(self):
        import torch
        from safetensors.torch import load_file, save_file

        from modelbuilder.helpers.nemotron import load_nemotron_checkpoint

        model = self._make_model(1)
        model_dir = self.get_model_dir("test_nemotron3_invalid_scales")
        self._write_modelopt_checkpoint(model, model_dir, 1)
        key = "backbone.layers.0.mixer.in_proj.weight_scale_2"
        for invalid in (float("nan"), -1.0, 0.0):
            with self.subTest(scale=invalid):
                with open(os.path.join(model_dir, "model.safetensors.index.json")) as stream:
                    filename = json.load(stream)["weight_map"][key]
                path = os.path.join(model_dir, filename)
                state = load_file(path)
                state[key] = torch.tensor([invalid])
                save_file(state, path)
                with self.assertRaisesRegex(ValueError, "must be finite and positive"):
                    load_nemotron_checkpoint(model_dir, model.config, model.config.num_hidden_layers)

    @hide_stdout()
    @requires_cuda()
    def test_prefill_decode_fp16_cuda(self):
        for variant in range(3):
            with self.subTest(model=MODEL_IDS[variant]):
                self._check_prefill_decode(variant, "fp16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_genai_generation(self):
        for variant in range(3):
            with self.subTest(model=MODEL_IDS[variant]):
                model, output_dir = self._export(variant)
                self.run_genai_generation_test(output_dir, model, model.config.vocab_size, model.config.eos_token_id)

    @hide_stdout()
    def test_onnx_light_generation(self):
        for variant in range(3):
            with self.subTest(model=MODEL_IDS[variant]):
                model, output_dir = self._export(variant)
                self.run_onnx_light_generation_test(output_dir, model, model.config.vocab_size, model.config.eos_token_id)

    @hide_stdout()
    @requires_genai()
    def test_modelopt_genai_generation(self):
        for variant in (1, 2):
            with self.subTest(model=MODEL_IDS[variant]):
                model, output_dir = self._export(variant, modelopt=True)
                self.run_genai_generation_test(output_dir, model, model.config.vocab_size, model.config.eos_token_id)

    @hide_stdout()
    def test_modelopt_onnx_light_generation(self):
        for variant in (1, 2):
            with self.subTest(model=MODEL_IDS[variant]):
                model, output_dir = self._export(variant, modelopt=True)
                self.run_onnx_light_generation_test(output_dir, model, model.config.vocab_size, model.config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
