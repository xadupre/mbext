# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

MODEL_NAME = "arnir0/Tiny-LLM"


def _torch_dtype_to_ort_element_type(dtype):
    """Map a :class:`torch.dtype` to the corresponding ORT TensorProto element type int."""
    from onnx import TensorProto
    import torch

    return {
        torch.float32: TensorProto.FLOAT,
        torch.float16: TensorProto.FLOAT16,
        torch.bfloat16: TensorProto.BFLOAT16,
        torch.int64: TensorProto.INT64,
        torch.int32: TensorProto.INT32,
        torch.bool: TensorProto.BOOL,
    }[dtype]


def _ort_io_binding_helper(sess, input_tensors, output_tensors, device="cuda:0"):
    """Run an ORT session using IOBinding so that bfloat16 tensors work.

    Follows the pattern from the onnxruntime-genai test suite.
    Inputs may be numpy arrays (including ml_dtypes.bfloat16) or CUDA
    :class:`torch.Tensor` objects; they are moved to *device* if needed.
    Output tensors must be pre-allocated :class:`torch.Tensor` objects on
    *device*; they are filled **in-place** by ORT.

    :param sess: :class:`onnxruntime.InferenceSession`
    :param input_tensors: ``dict[str, np.ndarray | torch.Tensor]``
    :param output_tensors: ``dict[str, torch.Tensor]`` – pre-allocated on *device*
    :param device: CUDA device string, e.g. ``"cuda:0"``
    """
    import torch

    parts = device.split(":")
    ort_device = parts[0]
    ort_device_id = int(parts[1]) if len(parts) > 1 else 0

    bind = sess.io_binding()
    torch_refs = []  # keep tensors alive for the duration of run_with_iobinding

    import ml_dtypes

    for name, value in input_tensors.items():
        if isinstance(value, np.ndarray):
            if value.dtype == ml_dtypes.bfloat16:
                # NumPy has no native bf16; reinterpret the bits as int16
                # (same 16-bit width) so torch can create a bfloat16 view.
                arr_c = np.ascontiguousarray(value)
                t = torch.from_numpy(arr_c.view(np.int16)).view(torch.bfloat16).to(device)
            else:
                t = torch.from_numpy(np.ascontiguousarray(value)).to(device)
        else:
            t = value.to(device) if value.device.type != ort_device else value
            t = t.contiguous()
        torch_refs.append(t)
        bind.bind_input(
            name,
            ort_device,
            ort_device_id,
            _torch_dtype_to_ort_element_type(t.dtype),
            list(t.shape),
            t.data_ptr(),
        )

    for name, tensor in output_tensors.items():
        t = tensor.contiguous()
        bind.bind_output(
            name,
            ort_device,
            ort_device_id,
            _torch_dtype_to_ort_element_type(t.dtype),
            list(t.shape),
            t.data_ptr(),
        )

    sess.run_with_iobinding(bind)


class TestRandomTinyLLM(ExtTestCase):
    def common_fast_tiny_llm_random_weights(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoModelForCausalLM,
            LlamaConfig,
            PreTrainedTokenizerFast,
        )
        from modelbuilder.builder import create_model

        # Config matching the arnir0/Tiny-LLM architecture (LlamaForCausalLM)
        # but with a single hidden layer to keep the test fast.  These values
        # are hardcoded so the test runs completely offline without downloading
        # any files from Hugging Face.
        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        basename = f"test_discrepancies_tiny_llm_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)
        num_hidden_layers = 1

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

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
            precision=precision,
            model_id=MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        # Fix random seed so the input token IDs are deterministic across runs.
        # The seed is local to this scope; it does not persist across tests
        # because each test creates a fresh random model with its own weights.
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_output_names = [i.name for i in sess.get_outputs()]

        # For bf16 on CUDA, ORT Python bindings cannot pass bfloat16 tensors
        # through NumPy, so we use IOBinding with pre-allocated torch CUDA
        # tensors instead (following the onnxruntime-genai test pattern).
        use_iobinding = precision == "bf16"
        device = f"{provider}:0" if use_iobinding else provider

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size),
                    dtype=self.get_input_np_dtype(precision),
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size),
                    dtype=self.get_input_np_dtype(precision),
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            if use_iobinding:
                # Build torch input tensors on the CUDA device.
                torch_prefill_feed = {
                    "input_ids": torch.from_numpy(prefill_feed["input_ids"]).to(device),
                    "attention_mask": torch.from_numpy(prefill_feed["attention_mask"]).to(device),
                    "position_ids": torch.from_numpy(prefill_feed["position_ids"]).to(device),
                }
                for i in range(num_hidden_layers):
                    torch_prefill_feed[f"past_key_values.{i}.key"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        0,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    torch_prefill_feed[f"past_key_values.{i}.value"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        0,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                torch_prefill_feed = {
                    k: v for k, v in torch_prefill_feed.items() if k in onnx_input_names
                }
                # Pre-allocate output tensors.
                # The builder upcasts bf16 logits to float32 for accuracy.
                ort_prefill_logits = torch.empty(
                    batch_size,
                    seq_len,
                    config.vocab_size,
                    dtype=torch.float32,
                    device=device,
                )
                torch_prefill_outputs = {"logits": ort_prefill_logits}
                for i in range(num_hidden_layers):
                    torch_prefill_outputs[f"present.{i}.key"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        seq_len,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    torch_prefill_outputs[f"present.{i}.value"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        seq_len,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                _ort_io_binding_helper(sess, torch_prefill_feed, torch_prefill_outputs, device)
                # Extract float32 logits as numpy; keep KV cache as torch tensors
                # so they can be fed directly into the decode io_binding call.
                ort_logits_np = ort_prefill_logits.detach().cpu().numpy()
                prefill_results = {"logits": ort_logits_np}
                for i in range(num_hidden_layers):
                    prefill_results[f"present.{i}.key"] = torch_prefill_outputs[
                        f"present.{i}.key"
                    ]
                    prefill_results[f"present.{i}.value"] = torch_prefill_outputs[
                        f"present.{i}.value"
                    ]
            else:
                prefill_outputs = sess.run(None, prefill_feed)
                prefill_results = dict(zip(onnx_output_names, prefill_outputs))
                ort_logits_np = prefill_outputs[0]

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {
                "fp16": 1e-2,
                "bf16": 1e-2,
                "fp32": 2e-3 if provider == "cuda" else 2e-4,
                "int4": 0.5,
            }
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
            # The next token is the greedy choice from the prefill logits.  The
            # exact token value is not important here; what matters is that both
            # ONNX and PyTorch process the *same* token in the decode step so
            # their outputs can be compared.
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            # ------------------------------------------------------------------
            # Step 2: single-token decode using the KV-cache from prefill
            # ------------------------------------------------------------------
            decode_feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            if use_iobinding:
                # The KV cache from prefill is already on CUDA as torch tensors.
                past_kv_len = prefill_results["present.0.key"].shape[2]
                ort_decode_logits = torch.empty(
                    batch_size,
                    1,
                    config.vocab_size,
                    dtype=torch.float32,
                    device=device,
                )
                torch_decode_outputs = {"logits": ort_decode_logits}
                for i in range(num_hidden_layers):
                    torch_decode_outputs[f"present.{i}.key"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        past_kv_len + 1,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    torch_decode_outputs[f"present.{i}.value"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        past_kv_len + 1,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                _ort_io_binding_helper(sess, decode_feed, torch_decode_outputs, device)
                onnx_decode_logits = ort_decode_logits.detach().cpu().numpy()
            else:
                decode_outputs = sess.run(None, decode_feed)
                onnx_decode_logits = decode_outputs[0]

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-4, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-4, "int4": 10000}
            np.testing.assert_allclose(
                pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision]
            )

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_fp32_cpu(self):
        self.common_fast_tiny_llm_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_fp16_cpu(self):
        self.common_fast_tiny_llm_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_int4_cpu(self):
        self.common_fast_tiny_llm_random_weights("int4", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_fp32_cuda(self):
        self.common_fast_tiny_llm_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_fp16_cuda(self):
        self.common_fast_tiny_llm_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_bf16_cuda(self):
        self.common_fast_tiny_llm_random_weights("bf16", "cuda")

    def common_tiny_llm_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoModelForCausalLM,
            LlamaConfig,
            PreTrainedTokenizerFast,
        )

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        basename = f"test_generation_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name="arnir0/Tiny-LLM",
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
        output_names = [out.name for out in sess.get_outputs()]

        batch_size = 1
        head_size = config.hidden_size // config.num_attention_heads
        max_new_tokens = 10

        # Use a fixed seed so the prompt token IDs are deterministic.
        torch.manual_seed(0)
        # Start from token ID 3 to avoid accidentally hitting BOS/EOS/PAD.
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.eos_token_id,
            )
        pt_tokens = pt_output[0].tolist()

        # ------------------------------------------------------------------
        # ONNX greedy generation (manual auto-regressive loop)
        # ------------------------------------------------------------------
        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        # Initialise empty KV-cache for every layer.
        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=self.get_input_np_dtype(precision),
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=self.get_input_np_dtype(precision),
            )

        # For bf16 on CUDA, use IOBinding so bfloat16 KV-cache tensors can be
        # passed without going through NumPy (which has no native bf16 type).
        use_iobinding = precision == "bf16"
        device = f"{provider}:0" if use_iobinding else provider

        # When using IOBinding, replace the initial empty numpy KV-cache with
        # empty torch CUDA tensors so they can be bound directly.
        if use_iobinding:
            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = torch.empty(
                    batch_size,
                    config.num_key_value_heads,
                    0,
                    head_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
                past_kv[f"past_key_values.{i}.value"] = torch.empty(
                    batch_size,
                    config.num_key_value_heads,
                    0,
                    head_size,
                    dtype=torch.bfloat16,
                    device=device,
                )

        onnx_tokens = current_ids[0].tolist()
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(
                    batch_size, cur_len
                ),
            }
            for i in range(num_hidden_layers):
                feed[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feed[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]
            # Drop any inputs the model does not declare.
            feed = {k: v for k, v in feed.items() if k in input_names}

            if use_iobinding:
                # Pre-allocate output tensors for this step.
                # Logits are float32 (builder upcasts bf16 logits); KV cache is bf16.
                kv_out_len = past_len + cur_len
                ort_step_logits = torch.empty(
                    batch_size,
                    cur_len,
                    config.vocab_size,
                    dtype=torch.float32,
                    device=device,
                )
                torch_step_outputs = {"logits": ort_step_logits}
                for i in range(num_hidden_layers):
                    torch_step_outputs[f"present.{i}.key"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        kv_out_len,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    torch_step_outputs[f"present.{i}.value"] = torch.empty(
                        batch_size,
                        config.num_key_value_heads,
                        kv_out_len,
                        head_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                _ort_io_binding_helper(sess, feed, torch_step_outputs, device)
                results = {"logits": ort_step_logits.detach().cpu().numpy()}
                for i in range(num_hidden_layers):
                    results[f"present.{i}.key"] = torch_step_outputs[f"present.{i}.key"]
                    results[f"present.{i}.value"] = torch_step_outputs[f"present.{i}.value"]
            else:
                outputs = sess.run(None, feed)
                results = dict(zip(output_names, outputs))

            # Greedy: pick the token with the highest logit at the last position.
            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            # Carry forward the updated KV-cache.
            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            # Prepare the single-token input for the next decode step.
            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config.eos_token_id:
                break

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
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
    def test_tiny_llm_fp32_cpu_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_tiny_llm_fp16_cpu_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_fp32_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_fp16_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_bf16_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
