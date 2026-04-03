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
        # The base make_model calls make_inputs_and_outputs() BEFORE
        # load_weights(), so we must determine the correct head_size from the
        # actual model weights BEFORE make_inputs_and_outputs() encodes the
        # wrong shape into the ONNX graph.
        #
        # Strategy: pre-load the weights here (which handles all cases –
        # local directories, HuggingFace downloads via model_name_or_path,
        # etc.), inspect the K-projection weight to get the real head_dim,
        # fix self.head_size if needed, then store the loaded model in
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
                break
        except Exception:
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
