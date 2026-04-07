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


class Ministral3TextModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def load_weights(self, input_path):
        # Mistral3ForConditionalGeneration (model_type="mistral3") is not
        # registered with AutoModelForCausalLM.  Load the full multimodal
        # model directly; make_model will find the embedded language-model
        # sub-modules (embed_tokens, MistralDecoderLayer, norm, lm_head)
        # via the standard module iteration already present in base.Model.
        if "ConditionalGeneration" in self.model_type:
            from transformers import Mistral3ForConditionalGeneration as HF_Mistral3VL

            return HF_Mistral3VL.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                trust_remote_code=self.hf_remote,
            )
        return super().load_weights(input_path)
