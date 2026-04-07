# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import os

import torch

from .base import Model, parse_hf_token


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


class _Ministral3VisionWrapper(torch.nn.Module):
    """Thin torch wrapper over the Pixtral vision tower + multimodal projector.

    The wrapper is designed to be exported to ONNX via :func:`torch.onnx.export`.
    It takes a single ``pixel_values`` tensor and returns the projected image
    features ready to be merged into the text-decoder's ``inputs_embeds``.

    ``image_sizes`` are inferred from the pixel-values shape at trace/export
    time, so the exported model is fixed to the square image resolution that
    was used during export (``vision_config.image_size``).

    .. note::
        This wrapper is intentionally limited to **batch_size = 1** (a single
        image per forward pass).  The ``Mistral3MultiModalProjector`` squeezes
        the batch dimension internally and operates on a flat sequence of
        patches, so supporting multiple images would require non-trivial
        changes to the projector interface.  For the typical generative
        inference case, images are processed one at a time before being
        concatenated into ``inputs_embeds``.
    """

    def __init__(self, vision_tower, projector, vision_feature_layer):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        self.vision_feature_layer = vision_feature_layer

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # image_sizes inferred from pixel_values tensor shape at trace time.
        # Batch size must be 1; see class docstring for details.
        batch_size = pixel_values.shape[0]
        h = pixel_values.shape[2]
        w = pixel_values.shape[3]
        image_sizes = [(h, w)] * batch_size

        outputs = self.vision_tower(
            pixel_values,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_dict=True,
        )

        if isinstance(self.vision_feature_layer, int):
            selected = outputs.hidden_states[self.vision_feature_layer]
        else:
            selected = torch.cat(
                [outputs.hidden_states[layer_idx] for layer_idx in self.vision_feature_layer],
                dim=-1,
            )

        # selected: [1, num_patches, hidden_size]
        # The projector expects a 2-D sequence [num_patches, hidden_size] so we
        # remove the batch dimension (batch_size == 1 enforced above).
        image_features = self.projector(selected.squeeze(0), image_sizes)
        return image_features


class Ministral3VisionEncoderModel:
    """Exports the Pixtral vision encoder + multimodal projector to ONNX.

    Unlike the text-decoder builder classes (which inherit from
    :class:`~modelbuilder.builders.base.Model` and build an ONNX graph
    manually), this class delegates to :func:`torch.onnx.export` because the
    Pixtral vision encoder contains operations (dynamic block-attention masks,
    2-D RoPE, patch merging) that are much simpler to export via the dynamo
    exporter than to replicate in ``onnx_ir``.

    The exported model has a single input ``pixel_values``
    (shape ``[1, num_channels, image_size, image_size]``) and a single output
    ``image_features`` (shape ``[num_merged_patches, text_hidden_size]``).
    The image resolution is fixed at ``vision_config.image_size`` × the same
    value, matching the default square crop used by Pixtral.
    """

    FILENAME = "vision_encoder.onnx"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.config = config
        self.vision_config = config.vision_config
        self.io_dtype = io_dtype
        self.cache_dir = cache_dir
        self.extra_options = extra_options
        self.filename = self.FILENAME
        self.hf_token = parse_hf_token(extra_options.get("hf_token", "true"))
        self.hf_remote = extra_options.get("hf_remote", True)
        self.model_name_or_path = config._name_or_path
        self._export_bytes = None  # in-memory ONNX after make_model()

    def _load_hf_model(self, input_path):
        from transformers import Mistral3ForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        # Force eager attention so that the ONNX exporter does not encounter
        # custom SDPA or Flash-Attention kernels it cannot handle.
        hf_cfg = copy.deepcopy(self.config)
        hf_cfg.vision_config._attn_implementation = "eager"
        # num_hidden_layers only applies to the text decoder; the full vision
        # encoder is always exported regardless of that option.
        return Mistral3ForConditionalGeneration.from_pretrained(
            src,
            config=hf_cfg,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
            **extra_kwargs,
        )

    def make_model(self, input_path):
        import io

        hf_model = self._load_hf_model(input_path)
        hf_model.eval()

        wrapper = _Ministral3VisionWrapper(
            hf_model.model.vision_tower,
            hf_model.model.multi_modal_projector,
            self.config.vision_feature_layer,
        )
        wrapper.eval()

        image_size = self.vision_config.image_size
        num_channels = self.vision_config.num_channels
        dummy_pixel_values = torch.zeros(1, num_channels, image_size, image_size)

        buf = io.BytesIO()
        torch.onnx.export(
            wrapper,
            (dummy_pixel_values,),
            buf,
            opset_version=18,
            input_names=["pixel_values"],
            output_names=["image_features"],
        )
        self._export_bytes = buf.getvalue()

    def save_model(self, out_dir):
        if self._export_bytes is None:
            raise RuntimeError("Call make_model() before save_model().")
        out_path = os.path.join(out_dir, self.filename)
        print(f"Saving vision encoder ONNX model in {out_dir}")
        with open(out_path, "wb") as f:
            f.write(self._export_bytes)


class Ministral3ConditionalGenerationModel:
    """Orchestrates exporting both the vision encoder and the text decoder for
    ``Mistral3ForConditionalGeneration`` (Ministral-3-3B-Instruct-2512).

    The exported artifacts are:

    * ``vision_encoder.onnx`` – Pixtral vision tower + multimodal projector
      (one fixed-resolution image in, projected embeddings out).
    * ``model.onnx`` – Mistral text decoder with ``exclude_embeds=True``
      (takes ``inputs_embeds`` from the vision encoder, outputs logits + KV cache).
    * ``genai_config.json`` – extended with a ``vision_encoder`` section.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Ministral3VisionEncoderModel(
            config, io_dtype, onnx_dtype, ep, cache_dir, extra_options
        )

        # --- Text decoder ---
        # Flatten text_config attributes onto the top-level config so that the
        # existing Ministral3TextModel constructor finds them (hidden_size, etc.).
        text_obj_config = copy.deepcopy(config)
        text_config = config.text_config
        for key in text_config:
            if not hasattr(text_obj_config, key) or key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "sliding_window",
                "vocab_size",
                "max_position_embeddings",
                "hidden_act",
                "rope_theta",
                "rope_scaling",
                "bos_token_id",
                "eos_token_id",
            ):
                setattr(text_obj_config, key, getattr(text_config, key))

        text_extra_options = dict(extra_options)
        text_extra_options["exclude_embeds"] = True

        self.text_model = Ministral3TextModel(
            text_obj_config, io_dtype, onnx_dtype, ep, cache_dir, text_extra_options
        )

    # ------------------------------------------------------------------
    # builder.py interface
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        print(
            "Building vision encoder (Pixtral + multimodal projector) for "
            "Mistral3ForConditionalGeneration..."
        )
        self.vision_encoder.make_model(input_path)
        print("Building text decoder for Mistral3ForConditionalGeneration...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Let the text model write genai_config.json first, then extend it
        # with a vision_encoder section.
        import json

        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        vision_cfg = self.vision_encoder.vision_config
        text_hidden_size = self.text_model.hidden_size
        image_size = vision_cfg.image_size
        patch_size = vision_cfg.patch_size
        spatial_merge_size = self.vision_encoder.config.spatial_merge_size
        num_patches_per_side = image_size // patch_size
        num_merged_patches = (num_patches_per_side**2) // (spatial_merge_size**2)

        genai_config["model"]["vision_encoder"] = {
            "filename": self.vision_encoder.filename,
            "hidden_size": vision_cfg.hidden_size,
            "image_size": image_size,
            "num_channels": vision_cfg.num_channels,
            "num_hidden_layers": vision_cfg.num_hidden_layers,
            "num_merged_patches": num_merged_patches,
            "patch_size": patch_size,
            "spatial_merge_size": spatial_merge_size,
            "text_hidden_size": text_hidden_size,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.text_model.save_processing(model_name_or_path, extra_kwargs, out_dir)
