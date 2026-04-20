# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import (
    ExtTestCase,
    has_transformers,
    hide_stdout,
    requires_cuda,
    requires_transformers,
    run_session_or_io_binding,
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


@requires_transformers("5")
class TestNemotronH(ExtTestCase):
    def common_fast_nemotron_h_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_discrepancies_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision, model_id=MODEL_NAME, experiment="forward", provider=provider, test=basename, input_type="text", kind="fast"
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = self.make_prefill_feed(
                input_ids.cpu().numpy(),
                num_hidden_layers,
                config.num_key_value_heads,
                head_size,
                self.get_input_np_dtype(precision),
                onnx_input_names,
            )

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                # use_cache=False avoids has_previous_state error when NemotronH
                # creates a DynamicCache internally (attention-only config).
                pt_prefill = model(input_ids, use_cache=False)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = self.make_decode_feed(
                next_token, seq_len, prefill_results, num_hidden_layers, batch_size=batch_size, onnx_input_names=onnx_input_names
            )

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                # NemotronH's _update_mamba_mask raises ValueError when
                # past_key_values is a DynamicCache (attention-only).
                # Run from scratch without cache to get the same logits.
                all_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(provider)], dim=1)
                pt_decode = model(all_ids, use_cache=False)
                pt_decode_logits = pt_decode.logits[:, -1:, :].detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_nemotron_h_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_generation_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.head_dim
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        with torch.no_grad():
            # use_cache=False avoids the has_previous_state error: NemotronH's
            # _update_mamba_mask raises ValueError when past_key_values is a
            # DynamicCache containing only attention layers (no Mamba layers).
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id, use_cache=False
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        onnx_tokens = self.run_onnx_greedy_generation(
            sess=sess,
            precision=precision,
            provider=provider,
            initial_ids=current_ids,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            max_new_tokens=max_new_tokens,
            input_names=input_names,
        )

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_fp32_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_fp16_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_int4_cpu(self):
        self.common_fast_nemotron_h_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_fp16_cuda(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_bf16_cuda(self):
        self.common_fast_nemotron_h_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_h_fp32_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_nemotron_h_fp16_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_fp16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_bf16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_h_fp32_cpu_genai_generate(self):
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        prefix = "test_nemotron_h_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)

        torch.manual_seed(0)
        batch_size = 1
        max_new_tokens = 5
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 4))
        prompt_len = prompt_ids.shape[1]

        if not has_transformers("5.5"):
            # The code is broken in transformers 5.5 for this model.
            # ValueError: `has_previous_state` can only be called on LinearAttention layers
            with torch.no_grad():
                pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
            pt_tokens = pt_output[0].tolist()

        og_model = og.Model(output_dir)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(prompt_ids.numpy().astype(np.int64))

        og_tokens = prompt_ids[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        if not has_transformers("5.5"):
            self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
