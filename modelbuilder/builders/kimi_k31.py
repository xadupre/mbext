# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Kimi K3.1 Pro export with explicit KDA/DSA state and MoonViT."""

import copy
import json
import os

import torch
from onnx_light.onnx import TensorProto

from ..helpers.kimi_k31 import KimiK31Checkpoint
from ..helpers.onnx_helper import get_default_onnx_opset
from ..models.kimi_k31 import KimiK31Config, KimiK31ForConditionalGeneration
from .base import Model
from .deepseek_v41 import DeepSeekV41Model, trace_v41_model


class _MediaGraph(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, patches, grid_shape):
        return self.module.encode_media(patches, grid_shape)


class KimiK31Model(DeepSeekV41Model):
    """Reuse component serialization, not DeepSeek architecture or cache aliases."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.kimi_config = KimiK31Config(**config.to_dict())
        text = copy.deepcopy(self.kimi_config.text_config)
        text.architectures = self.kimi_config.architectures
        text._name_or_path = config._name_or_path
        # Source MXFP4 is decoded before the ordinary optional ONNX quantizer.
        if hasattr(text, "quantization_config"):
            del text.quantization_config
        Model.__init__(self, text, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if ep not in {"cpu", "cuda", "any"}:
            raise ValueError("Kimi K3.1 supports cpu, cuda and any exports.")
        unsupported = (
            "exclude_embeds",
            "exclude_lm_head",
            "adapter_path",
            "reuse_downloaded_weights",
            "shared_embeddings",
            "include_hidden_states",
            "prune_lm_head",
        )
        if any(extra_options.get(key) for key in unsupported):
            raise ValueError("Kimi K3.1 requires complete, independent embeddings and weights; adapters and weight reuse are unsupported.")
        if self.num_layers != text.num_hidden_layers:
            raise ValueError("Change the Kimi text config and attention schedule together, not num_hidden_layers extra_options.")
        if self.shared_embeddings:
            raise ValueError("Kimi K3.1 requires independent embeddings and LM-head weights.")
        self.quant_type = None
        self._components = {}
        self._active_component = "decoder"
        self._manifest = None
        # This exporter traces through TorchScript, whose graph serializer
        # supports fewer opsets than the native graph-construction builders.
        self._opset = int(extra_options.get("onnx_opset", min(23, get_default_onnx_opset())))
        if not 20 <= self._opset <= 23:
            raise ValueError("Kimi's TorchScript exporter supports ONNX opsets 20 through 23.")
        if self._opset < 21 and self.quant_attrs["use_qdq"]:
            raise ValueError("Kimi INT4 QDQ requires ONNX opset 21 or newer.")

    def make_model(self, input_path):
        directory = input_path if os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(directory):
            from huggingface_hub import snapshot_download

            directory = snapshot_download(
                directory, cache_dir=self.cache_dir, token=self.hf_token, allow_patterns=["*.json", "*.safetensors"]
            )
        quant = getattr(self.kimi_config.text_config, "quantization_config", None)
        if quant and (quant.get("quant_method"), quant.get("format")) != ("compressed-tensors", "mxfp4-pack-quantized"):
            raise ValueError("Only native floating-point and compressed-tensors MXFP4 K3.1 checkpoints are supported.")
        with torch.device("meta"):
            module = KimiK31ForConditionalGeneration(self.kimi_config)
        dtype = {TensorProto.FLOAT: torch.float32, TensorProto.FLOAT16: torch.float16, TensorProto.BFLOAT16: torch.bfloat16}[self.io_dtype]
        KimiK31Checkpoint(directory).load(module, dtype=dtype)
        self.export_module(module.eval())

    @torch.no_grad()
    def export_module(self, module):
        text = self.kimi_config.text_config
        state = module.initial_state(2)
        dtype = module.language_model.model.embed_tokens.weight.dtype
        ids = torch.zeros(2, 1, dtype=torch.int64)
        positions = torch.zeros_like(ids)
        types = torch.full_like(ids, -1)
        embeds = torch.zeros(2, 1, text.hidden_size, dtype=dtype)
        _, *seeded = module(ids, positions, types, embeds, *state)
        names = ["input_ids", "position_ids", "token_types", "inputs_embeds", *module.state_names]
        outputs = ["logits", *[f"present.{name}" for name in module.state_names]]
        axes = {name: {0: "batch_size"} for name in names + outputs}
        for name in module.state_names:
            if name == "positions" or name.endswith(("_latent", "_shared_key", "_index_key")):
                axes[name][1] = "past_length"
                axes[f"present.{name}"][1] = "total_length"
        graph = trace_v41_model(module, (ids, positions + 1, types, embeds, *seeded), names, outputs, axes, opset_version=self._opset)
        graph.graph.name = "kimi_k31_decoder_step"
        self._components["decoder"] = graph
        self._manifest = {
            "format": "kimi_k31",
            "version": 1,
            "decoder": self.filename,
            "genai_supported": False,
            "genai_issue": "https://github.com/xadupre/mbext/issues/604",
            "hidden_size": text.hidden_size,
            "vocab_size": text.vocab_size,
            "eos_token_id": (
                self.kimi_config.eos_token_id
                if self.kimi_config.eos_token_id is not None
                else getattr(text, "eos_token_id", None)
            ),
            "context_length": text.max_position_embeddings,
            "position_state": "positions",
            "io_dtype": str(dtype).removeprefix("torch."),
            "media_placeholder_token_id": self.kimi_config.media_placeholder_token_id,
            "state": [
                dict(
                    input=name,
                    output=f"present.{name}",
                    shape=list(value.shape),
                    dtype=str(value.dtype).removeprefix("torch."),
                    fill=-1 if name == "positions" else 0,
                )
                for name, value in zip(module.state_names, state)
            ],
        }
        vision = self.kimi_config.vision_config
        if vision is not None:
            ph, pw = vision["merge_kernel_size"]
            grid = torch.tensor([2, ph * 2, pw * 3], dtype=torch.int64)
            patches = torch.zeros(2 * ph * 2 * pw * 3, 3, vision["patch_size"], vision["patch_size"], dtype=dtype)
            graph = trace_v41_model(
                _MediaGraph(module),
                (patches, grid),
                ["patches", "grid_shape"],
                ["image_features"],
                {"patches": {0: "num_patches"}, "image_features": {0: "num_media_tokens"}},
                opset_version=self._opset,
            )
            graph.graph.name = "kimi_k31_moonvit"
            self._components["vision"] = graph
            self._manifest["vision"] = os.path.splitext(self.filename)[0] + ".vision.onnx"
            self._manifest["vision_config"] = vision
        if (
            self.onnx_dtype in {TensorProto.INT4, TensorProto.UINT4}
            and not self.quant_attrs["int4"]["quantize_moe_router"]
        ):
            excluded = self.quant_attrs["int4"]["nodes_to_exclude"]
            excluded.extend(
                node.name
                for node in self._components["decoder"].graph.node
                if node.op_type == "MatMul"
                and "block_sparse_moe" in node.name
                and ("gate" in node.name or "router" in node.name)
                and node.name not in excluded
            )

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        if self._manifest is None:
            raise ValueError("Kimi K3.1 config-only export needs explicit recurrent-state metadata.")
        for filename in ("genai_config.json", "pipeline.onnx"):
            path = os.path.join(out_dir, filename)
            if os.path.exists(path):
                os.remove(path)
        with open(os.path.join(out_dir, "kimi_k31_config.json"), "w") as stream:
            json.dump(self._manifest, stream, indent=2)
        print("Kimi K3.1 uses KimiK31Session, with sequential prefill and explicit recurrent state; native GenAI is blocked by #604.")
        return self._manifest
