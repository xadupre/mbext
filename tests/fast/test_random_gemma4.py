# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

MODEL_NAME = "google/gemma-4-E4B-it"


@requires_transformers("5.6")
class TestRandomGemma4(ExtTestCase):
    @staticmethod
    def _make_config(num_hidden_layers=2):
        """Return a minimal Gemma4TextConfig for offline testing.

        Uses equal head_dim and global_head_dim (both 64) to keep the
        KV-cache shapes uniform across layer types.  PLE and shared-KV
        features are disabled so the export exercises only the core
        sliding/full-attention and v_norm paths.

        ``layer_types`` is set explicitly so that the last layer is
        ``"full_attention"`` (required by Gemma4) without relying on
        the config's auto-correction that emits a WARNING.
        """
        from transformers import Gemma4TextConfig

        # Build layer_types: (num_hidden_layers - 1) sliding + 1 full.
        layer_types = ["sliding_attention"] * (num_hidden_layers - 1) + ["full_attention"]

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=64,  # keep head_dim uniform across layer types
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=layer_types,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
            # Disable optional features not yet exported
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=False,
        )

    @staticmethod
    def _make_shared_kv_config(num_hidden_layers=4, num_kv_shared=2):
        """Return a Gemma4TextConfig with ``num_kv_shared_layers > 0``.

        Layer layout: [sliding, full, sliding, full, ...]
        The last ``num_kv_shared`` layers share K/V with the last non-shared
        layer of the same type.  Both non-shared and shared sections must
        contain at least one sliding and one full-attention layer so that each
        shared layer has a valid donor.

        ``num_hidden_layers`` must be >= 4 and ``num_kv_shared`` must satisfy
        ``2 <= num_kv_shared <= num_hidden_layers - 2`` so that at least one
        non-shared sliding *and* one non-shared full-attention layer always exist.
        """
        if num_hidden_layers < 4:
            raise ValueError(f"num_hidden_layers must be >= 4, got {num_hidden_layers}")
        if num_kv_shared < 2 or num_kv_shared > num_hidden_layers - 2:
            raise ValueError(
                f"num_kv_shared must be in [2, num_hidden_layers - 2], "
                f"got num_kv_shared={num_kv_shared}, num_hidden_layers={num_hidden_layers}"
            )
        from transformers import Gemma4TextConfig

        layer_types = []
        for i in range(num_hidden_layers):
            layer_types.append("sliding_attention" if i % 2 == 0 else "full_attention")

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=layer_types,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=num_kv_shared,
            enable_moe_block=False,
        )

    def common_fast_gemma4_shared_kv_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_shared_kv_config()
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma4_shared_kv_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": config.num_hidden_layers},
            atol={"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 1.5},
        )

    def common_gemma4_shared_kv_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_shared_kv_config()
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma4_shared_kv_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": config.num_hidden_layers},
        )

    @staticmethod
    def _make_ple_config(num_hidden_layers=2):
        """Return a minimal Gemma4TextConfig with Per-Layer Embeddings (PLE) enabled.

        Both ``vocab_size`` and ``vocab_size_per_layer_input`` are set to 1000
        so that all token IDs generated during tests (by ``torch.randint`` or
        ``model.generate()``) are valid for both embedding tables.  1000 is
        chosen to be comfortably above ``num_kv_heads * head_dim = 256`` and
        below the default 32000, keeping the test fast while avoiding false
        positives in the ``has_lm_head`` check (``out_features == vocab_size``).
        """
        from transformers import Gemma4TextConfig

        layer_types = ["sliding_attention"] * (num_hidden_layers - 1) + ["full_attention"]

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=layer_types,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            sliding_window=512,
            # Match vocab_size and vocab_size_per_layer_input so that any
            # generated token ID is valid for both embedding tables.
            # 1000 > num_kv_heads * head_dim (= 256) avoids has_lm_head false positives.
            vocab_size=1000,
            hidden_size_per_layer_input=64,
            vocab_size_per_layer_input=1000,
            num_kv_shared_layers=0,
            enable_moe_block=False,
        )

    def common_fast_gemma4_ple_random_weights(self, precision, provider):
        """Export and run a random-weight Gemma4 model with PLE enabled."""
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_ple_config()
        num_hidden_layers = config.num_hidden_layers

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma4_ple_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            # vocab_size matches vocab_size_per_layer_input so random IDs stay within
            # both embedding table bounds (main vocab and per-layer vocab).
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            atol={"fp16": 2e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 1.5},
        )

    def common_gemma4_ple_greedy_generation(self, precision, provider):
        """Greedy-generation check for a Gemma4 model with PLE enabled."""
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_ple_config()
        num_hidden_layers = config.num_hidden_layers

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma4_ple_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            # vocab_size == vocab_size_per_layer_input: all generated tokens are
            # in-range for both embedding tables.
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    def common_fast_gemma4_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        num_hidden_layers = config.num_hidden_layers

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma4_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            # Use a relaxed int4 tolerance: 2 layers with random weights can exceed 0.5
            atol={"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 1.5},
        )

    def common_gemma4_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        num_hidden_layers = config.num_hidden_layers

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma4_{precision}_{provider}",
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
    def test_fast_discrepancy_gemma4_fp32_cpu(self):
        self.common_fast_gemma4_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_fp16_cpu(self):
        self.common_fast_gemma4_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_int4_cpu(self):
        self.common_fast_gemma4_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma4_fp16_cuda(self):
        self.common_fast_gemma4_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gemma4_fp32_cpu_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma4_fp16_cpu_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_gemma4_fp16_cuda_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma4_bf16_cuda_greedy_generation(self):
        self.common_gemma4_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_gemma4_num_kv_shared_layers(self):
        """Gemma4 with num_kv_shared_layers > 0 should export without errors.

        Shared-KV layers (the last ``num_kv_shared_layers`` layers) have no
        k_proj/v_proj/k_norm/v_norm weights and reuse the donor layer's K/V.
        This test verifies that ``create_model`` completes successfully for such
        a configuration and produces a valid ONNX model.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        # 4 layers: [sliding, full, sliding, full]
        # Non-shared layers: [0, 1] (sliding + full)
        # Shared layers:    [2, 3] (sliding reuses layer 0 as donor; full reuses layer 1 as donor)
        # first_kv_shared_layer_idx = 4 - 2 = 2
        # Layer 2 (sliding): last non-shared sliding = layer 0 → donor = 0
        # Layer 3 (full):    last non-shared full    = layer 1 → donor = 1
        num_hidden_layers = 4
        num_kv_shared = 2
        config = self._make_config(num_hidden_layers=num_hidden_layers)
        # Override layer_types so that both non-shared and shared sections
        # contain at least one sliding and one full-attention layer.
        config.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        config.num_kv_shared_layers = num_kv_shared

        prefix = "test_gemma4_num_kv_shared_layers"
        model_dir = self.get_model_dir(prefix, clean=True)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=True)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

    @hide_stdout()
    def test_fast_discrepancy_gemma4_shared_kv_fp32_cpu(self):
        self.common_fast_gemma4_shared_kv_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_shared_kv_fp16_cpu(self):
        self.common_fast_gemma4_shared_kv_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_gemma4_shared_kv_fp32_cpu_greedy_generation(self):
        self.common_gemma4_shared_kv_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma4_shared_kv_fp16_cpu_greedy_generation(self):
        self.common_gemma4_shared_kv_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_gemma4_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        prefix = "test_gemma4_fp32_cpu_genai_generate"
        config = self._make_config(num_hidden_layers=2)

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)

    @hide_stdout()
    def test_gemma4_autoround(self):
        """Gemma4 model with AutoRound quantization config should export correctly.

        This test simulates a Gemma4 model (Intel/gemma-4-31B-it-int4-AutoRound) by
        saving a random float32 Gemma4ForCausalLM checkpoint and then patching its
        config.json to add ``quantization_config.quant_method = "autoround"``.

        The builder must route such models through the AutoRound loading path
        (``Gemma4Model.load_weights`` → ``AutoModelForCausalLM.from_pretrained``)
        rather than the default ``QuantModel.from_pretrained`` path, which does
        not understand the AutoRound quant_method.  Transformers treats
        ``"autoround"`` as an unrecognised quantization type and simply loads
        the float32 safetensors, allowing the builder to export a valid ONNX
        model.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_config()
        prefix = "test_gemma4_autoround"
        model_dir = self.get_model_dir(prefix, clean=True)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        # Inject a fake AutoRound quantization_config into the saved config.json.
        # The quant_method value "autoround" is not recognised by transformers, so
        # it falls back to loading plain float32 weights – which is exactly what the
        # test model contains.  In production the model directory would contain
        # GPTQ-packed int4 weights and the auto-round library would handle
        # dequantisation transparently.
        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path) as f:
            cfg_data = json.load(f)
        cfg_data["quantization_config"] = {
            "quant_method": "autoround",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f, indent=4)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=True)

        create_model(
            model_name=INTEL_AUTOROUND_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

    @hide_stdout()
    def test_gemma4_autoround_int4(self):
        """AutoRound-tagged Gemma4 can be exported with precision=int4.

        This test re-uses the same ``"autoround"`` (unrecognised by transformers)
        trick as :meth:`test_gemma4_autoround` so that the model loads as plain
        float32.  The builder then quantises the float MatMul nodes to
        ``MatMulNBits`` via :class:`onnxruntime.quantization.MatMulNBitsQuantizer`
        in the usual way.

        The test verifies two things:
        1. The ``quant_type in _AUTOROUND_QUANT_ALIASES`` routing in
           :meth:`Gemma4Model.load_weights` does not break the ``precision=int4``
           export path.
        2. An ``int4`` ONNX model is correctly produced.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_config()
        prefix = "test_gemma4_autoround_int4"
        model_dir = self.get_model_dir(prefix, clean=True)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        cfg_path = os.path.join(model_dir, "config.json")
        with open(cfg_path) as f:
            cfg_data = json.load(f)
        cfg_data["quantization_config"] = {
            "quant_method": "autoround",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f, indent=4)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=True)

        create_model(
            model_name=INTEL_AUTOROUND_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="int4",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

    @hide_stdout()
    def test_fast_discrepancy_gemma4_ple_fp32_cpu(self):
        """Gemma4 with Per-Layer Embeddings (PLE) – fp32 random-weight discrepancy check."""
        self.common_fast_gemma4_ple_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_ple_fp16_cpu(self):
        """Gemma4 with Per-Layer Embeddings (PLE) – fp16 random-weight discrepancy check."""
        self.common_fast_gemma4_ple_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_gemma4_ple_fp32_cpu_greedy_generation(self):
        """Gemma4 with Per-Layer Embeddings (PLE) – fp32 greedy generation check."""
        self.common_gemma4_ple_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma4_ple_fp16_cpu_greedy_generation(self):
        """Gemma4 with Per-Layer Embeddings (PLE) – fp16 greedy generation check."""
        self.common_gemma4_ple_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_gemma4_ple_fp32_cpu_genai_generate(self):
        """Gemma4 with Per-Layer Embeddings (PLE) – fp32 ORT-GenAI generation check."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        prefix = "test_gemma4_ple_fp32_cpu_genai_generate"
        config = self._make_ple_config(num_hidden_layers=2)

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)

    @staticmethod
    def _make_global_head_dim_config():
        """Return a Gemma4TextConfig where ``global_head_dim != head_dim``.

        Uses ``head_dim=64`` (sliding-attention) and ``global_head_dim=128``
        (full-attention) with a minimal 2-layer architecture so that the
        KV-cache shape mismatch between layer types is exercised.
        """
        from transformers import Gemma4TextConfig

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=128,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=["sliding_attention", "full_attention"],
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=2,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=False,
        )

    @hide_stdout()
    def test_gemma4_genai_config_head_size_global_head_dim(self):
        """genai_config.json head_size must use global_head_dim when it differs from head_dim.

        Regression test for the bug where ``head_size`` in genai_config.json
        reflected the sliding-attention ``head_dim`` instead of ``global_head_dim``
        when ``global_head_dim != head_dim``.

        Also runs a prefill/decode discrepancy check (ONNX vs PyTorch) using the
        correct per-layer KV-cache head sizes (``global_head_dim`` for
        ``full_attention`` layers, ``head_dim`` for ``sliding_attention`` layers).
        """
        import json
        import os

        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_global_head_dim_config()
        sliding_head_size = config.head_dim  # 64
        global_head_size = config.global_head_dim  # 128
        assert global_head_size != sliding_head_size
        layer_types = config.layer_types

        prefix = "test_gemma4_genai_config_head_size_global_head_dim"
        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        # --- genai_config.json head_size check ---
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path) as f:
            genai_config = json.load(f)

        actual_head_size = genai_config["model"]["decoder"]["head_size"]
        self.assertEqual(
            actual_head_size,
            global_head_size,
            f"genai_config.json head_size should be global_head_size ({global_head_size}), got {actual_head_size}",
        )

        # --- Discrepancy check: ONNX vs PyTorch (prefill + decode) ---
        # The model has layers with different KV-cache head sizes:
        #   sliding_attention → head_size = sliding_head_size (head_dim)
        #   full_attention    → head_size = global_head_size (global_head_dim)
        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path, provider="cpu")
        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        output_names = [o.name for o in sess.get_outputs()]

        # Per-layer head sizes based on layer type
        layer_head_sizes = [global_head_size if lt == "full_attention" else sliding_head_size for lt in layer_types]

        batch_size, seq_len = 1, 5
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        prefill_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for layer_idx, head_size in enumerate(layer_head_sizes):
            prefill_feed[f"past_key_values.{layer_idx}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
            prefill_feed[f"past_key_values.{layer_idx}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32
            )
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

        prefill_results = dict(zip(output_names, sess.run(None, prefill_feed)))
        ort_prefill_logits = prefill_results["logits"]

        with torch.no_grad():
            pt_prefill = model(input_ids)
        pt_prefill_logits = pt_prefill.logits.detach().cpu().numpy()

        with self.subTest(step="prefill"):
            np.testing.assert_allclose(pt_prefill_logits, ort_prefill_logits, atol=1e-3, rtol=1e-3)

        # Decode step: reuse present KV tensors from prefill as past KV
        next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))
        decode_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            "position_ids": np.array([[seq_len]], dtype=np.int64),
        }
        for layer_idx in range(len(layer_head_sizes)):
            decode_feed[f"past_key_values.{layer_idx}.key"] = prefill_results[f"present.{layer_idx}.key"]
            decode_feed[f"past_key_values.{layer_idx}.value"] = prefill_results[f"present.{layer_idx}.value"]
        decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

        decode_results = dict(zip(output_names, sess.run(None, decode_feed)))
        ort_decode_logits = decode_results["logits"]

        with torch.no_grad():
            pt_decode = model(torch.tensor([[next_token]], dtype=torch.long), past_key_values=pt_prefill.past_key_values)
        pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

        with self.subTest(step="decode"):
            np.testing.assert_allclose(pt_decode_logits, ort_decode_logits, atol=1e-3, rtol=1e-3)

    @hide_stdout()
    @requires_genai()
    def test_gemma4_genai_config_head_size_global_head_dim_genai(self):
        """ORT-GenAI generation for Gemma4 with a non-default ``global_head_dim``.

        Uses ``head_dim == global_head_dim = 128`` so that ORT-GenAI can allocate
        uniform KV-cache buffers (ORT-GenAI uses a single ``head_size`` value
        from genai_config.json for all layers and does not support per-layer
        head sizes).  The companion discrepancy test covers the
        ``head_dim != global_head_dim`` scenario for the ONNX model.
        """
        import torch
        from transformers import AutoModelForCausalLM, Gemma4TextConfig

        from modelbuilder.builder import create_model

        # Use head_dim == global_head_dim = 128 so ORT-GenAI (which allocates
        # a uniform KV cache of shape [..., head_size]) works for all layers.
        config = Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=128,
            global_head_dim=128,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=["sliding_attention", "full_attention"],
            max_position_embeddings=2048,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=False,
        )

        prefix = "test_gemma4_genai_config_head_size_global_head_dim_genai"
        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
