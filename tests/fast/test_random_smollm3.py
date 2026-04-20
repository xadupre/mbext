# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


class TestSmolLM3(ExtTestCase):
    def common_fast_smollm3_random_weights(self, precision, provider):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model

        # num_hidden_layers=4 is required so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        basename = f"test_discrepancies_smollm3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=SMOLLM3_MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path, provider=provider)

        self.run_prefill_and_decode_check(
            model=model,
            sess=sess,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            precision=precision,
            provider=provider,
            log_data=log_data,
        )

    def common_smollm3_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model

        # num_hidden_layers=4 is required so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        basename = f"test_generation_smollm3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        log_data = dict(
            precision=precision,
            model_id=SMOLLM3_MODEL_NAME,
            experiment="generate",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )
        self.run_greedy_generation_check(
            model=model,
            sess=sess,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            precision=precision,
            provider=provider,
            log_data=log_data,
        )

    @hide_stdout()
    def test_smollm3_fp32_cpu_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp32", "cpu")

    @unittest.skip("issue")
    @hide_stdout()
    def test_smollm3_fp16_cpu_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_smollm3_fp32_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_smollm3_fp16_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_smollm3_bf16_cuda_greedy_generation(self):
        self.common_smollm3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp32_cpu(self):
        self.common_fast_smollm3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp16_cpu(self):
        self.common_fast_smollm3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_int4_cpu(self):
        self.common_fast_smollm3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_smollm3_fp16_cuda(self):
        self.common_fast_smollm3_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_smollm3_fp32_cpu_genai_generate(self):
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model

        prefix = "test_smollm3_fp32_cpu_genai_generate"
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
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

        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
