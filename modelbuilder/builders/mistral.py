# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .base import Model


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralNeMoModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_model(self, input_path):
        # Mistral-Nemo-Instruct uses head_dim=128 even though
        # hidden_size / num_attention_heads = 160.  When the model config.json
        # does not carry an explicit ``head_dim`` field the base class falls
        # back to the division and produces the wrong value (160).
        #
        # CRITICAL: self.input_shapes and self.output_shapes are built inside
        # Model.__init__ with the INTEGER value of self.head_size baked into
        # the KV-cache shape lists at construction time.  make_inputs_and_outputs()
        # reads those pre-built lists, so updating self.head_size alone after
        # __init__ is NOT enough — the cached shape lists must also be patched.
        #
        # Strategy: pre-load the weights here (which handles all cases –
        # local directories, HuggingFace downloads via model_name_or_path,
        # etc.), inspect the K-projection weight to get the real head_dim,
        # fix self.head_size if needed, also patch self.input_shapes and
        # self.output_shapes, then store the loaded model in
        # ``_preloaded_weights``.  The ``load_weights`` override below
        # returns the cached model on the next call so the weights are only
        # loaded once.
        try:
            self._preloaded_weights = self.load_weights(input_path)
            for module in self._preloaded_weights.modules():
                k_proj = getattr(module, "k_proj", None)
                if k_proj is None:
                    continue
                weight = getattr(k_proj, "weight", None)
                if weight is None:
                    continue
                # k_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
                actual_head_size = weight.shape[0] // self.num_kv_heads
                if actual_head_size != self.head_size:
                    self.head_size = actual_head_size
                    # Patch the KV-cache shape lists that were baked into
                    # self.input_shapes / self.output_shapes at __init__ time.
                    # Each shape list is [batch, num_kv_heads, seq, head_size]
                    # so head_size is always the last element (index 3).
                    for key in ("past_key_values.key", "past_key_values.value"):
                        self.input_shapes[key][3] = actual_head_size
                    for key in ("present.key", "present.value"):
                        self.output_shapes[key][3] = actual_head_size
                break
        # The pre-load is best-effort: if it fails for any reason (missing
        # files, unsupported model format, network error, attribute not found,
        # etc.) we clean up and let the base make_model proceed.  The base
        # make_model will re-attempt the load and may raise its own error with
        # a clearer message.  Note that Exception does not catch
        # KeyboardInterrupt or SystemExit.
        except Exception:  # noqa: BLE001
            if hasattr(self, "_preloaded_weights"):
                del self._preloaded_weights

        super().make_model(input_path)

    def load_weights(self, input_path):
        # Return the pre-loaded model when available so that the full model
        # is loaded only once (the pre-load in make_model plus this call from
        # base.make_model would otherwise load it twice).
        if hasattr(self, "_preloaded_weights"):
            weights = self._preloaded_weights
            del self._preloaded_weights
            return weights
        return super().load_weights(input_path)
