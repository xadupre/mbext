# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Export DeepSeek-V4.1's heterogeneous recurrent state without a V3 alias."""

import copy
import json
import math
import os

import torch
from onnx_light.onnx import ModelProto, TensorProto, helper, numpy_helper

from ..helpers.deepseek_v41 import DeepseekV41Checkpoint
from ..helpers.onnx_helper import DEFAULT_ONNX_IR_VERSION, torch_tensor_to_numpy
from .base import Model


def _json_state(value):
    if isinstance(value, list):
        return [_json_state(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def trace_v41_model(module, args, input_names, output_names, dynamic_axes, *, opset_version=20):
    """Trace operations in memory, attaching weights through native ONNX APIs."""
    from torch.onnx._internal.torchscript_exporter import utils
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

    # Stop at the native graph serializer: torch.onnx.export's file-writing
    # wrapper imports standard ONNX. All model I/O here belongs to onnx-light.
    settings = {
        key: getattr(GLOBALS, key)
        for key in ("in_onnx_export", "export_onnx_opset_version", "operator_export_type", "onnx_shape_inference")
    }
    if settings["in_onnx_export"]:
        raise RuntimeError("Cannot nest DeepSeek graph tracing inside another export.")
    operator_type = torch.onnx.OperatorExportTypes.ONNX
    utils._trigger_symbolic_function_registration()
    GLOBALS.in_onnx_export = True
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_type
    GLOBALS.onnx_shape_inference = True
    try:
        utils._setup_trace_module_map(module, False)
        with utils.exporter_context(module, torch.onnx.TrainingMode.EVAL, False):
            graph, _, _ = utils._model_to_graph(
                module, args, input_names=input_names, output_names=output_names, do_constant_folding=False, dynamic_axes=dynamic_axes
            )
            proto, _, _, _ = graph._export_onnx({}, opset_version, dynamic_axes, False, operator_type, True, True, {}, True, "", {})
    finally:
        utils._reset_trace_module_map()
        for key, value in settings.items():
            setattr(GLOBALS, key, value)
    model = ModelProto()
    model.ParseFromString(proto)
    parameters = dict(module.named_parameters())
    parameters.update(module.named_buffers())
    remaining = set(input_names)
    graph_inputs = []
    for value in model.graph.input:
        cloned = type(value)()
        cloned.CopyFrom(value)
        graph_inputs.append(cloned)
    unknown = {value.name for value in graph_inputs} - parameters.keys() - remaining
    if unknown:
        raise ValueError(f"Unmapped traced DeepSeek inputs: {sorted(unknown)}.")
    transposed = {}
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Transpose" and str(node.input[0]) in parameters:
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            permutation = attributes.get("perm")
            value = parameters[str(node.input[0])]
            if permutation is None:
                permutation = list(reversed(range(value.ndim)))
            transposed[str(node.output[0])] = value.permute(*permutation)
        else:
            cloned = type(node)()
            cloned.CopyFrom(node)
            nodes.append(cloned)
    consumed = {name for node in nodes for name in node.input}
    consumed.update(value.name for value in model.graph.output)
    initializers = []
    for name, value in {**parameters, **transposed}.items():
        if name in consumed:
            initializers.append(numpy_helper.from_array(torch_tensor_to_numpy(value), name=name))
    graph = helper.make_graph(
        nodes,
        "deepseek_v41",
        [value for value in graph_inputs if value.name in remaining],
        list(model.graph.output),
        initializer=initializers,
    )
    model.graph.CopyFrom(graph)
    model.ir_version = DEFAULT_ONNX_IR_VERSION
    model.producer_name = "mbext"
    return model


class _VisionGraph(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, patches, grid_shape):
        return (
            self.module.encode_image(patches, grid_shape[0], grid_shape[1]),
            self.module.image_start.clone(),
            self.module.image_end.clone(),
            self.module.image_newline.clone(),
        )


class DeepSeekV41Model(Model):
    """Separate vision and stateful language graphs for the V4.1 architecture.

    The state manifest is consumed by ``DeepseekV41Session``. GenAI's ordinary
    decoder cache manager cannot bind these heterogeneous tensors (see #604).
    """

    supports_genai = False

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.v41_config = copy.deepcopy(config)
        text = copy.deepcopy(config.text_config)
        text.intermediate_size = text.moe_intermediate_size
        text.architectures = list(config.architectures)
        text._name_or_path = config._name_or_path
        super().__init__(text, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if self.num_layers != text.num_hidden_layers:
            raise ValueError(
                "DeepSeek-V4.1 layer counts must be changed in text_config together " "with its compressed-attention sharing schedule."
            )
        if extra_options.get("reuse_downloaded_weights", False):
            raise ValueError("DeepSeek-V4.1 FP8/FP4 decoding cannot reuse the original " "safetensors byte ranges.")
        if ep not in {"cpu", "cuda", "any"}:
            raise ValueError("DeepSeek-V4.1 supports cpu, cuda and any exports.")
        if extra_options.get("exclude_embeds") or extra_options.get("exclude_lm_head"):
            raise ValueError("DeepSeek-V4.1 exports require embeddings and the LM head.")
        if "adapter_path" in extra_options:
            raise ValueError("DeepSeek-V4.1 adapter export is not supported.")
        self.quant_type = None
        self._components = {}
        self._active_component = "decoder"
        self._manifest = None

    def to_model_proto(self):
        model = self._components[self._active_component]
        if self.ep == "any":
            from .portable import validate_portable_model

            validate_portable_model(model)
        return model

    def make_model(self, input_path):
        from transformers import AutoTokenizer

        from ..models.deepseek_v41 import DeepseekV41ForCausalLM

        directory = input_path if os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(directory):
            from huggingface_hub import snapshot_download

            directory = snapshot_download(
                directory, cache_dir=self.cache_dir, token=self.hf_token, allow_patterns=["*.json", "*.safetensors"]
            )
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        with torch.device("meta"):
            module = DeepseekV41ForCausalLM(self.v41_config, tokenizer=tokenizer)
        quantization = getattr(self.v41_config, "quantization_config", {}) or {}
        weights = DeepseekV41Checkpoint(directory, block_size=quantization.get("weight_block_size", (32, 32)))
        dtype = {TensorProto.FLOAT: torch.float32, TensorProto.FLOAT16: torch.float16, TensorProto.BFLOAT16: torch.bfloat16}[self.io_dtype]
        weights.load(module, dtype=dtype)
        self.export_module(module)

    @torch.no_grad()
    def export_module(self, module):
        """Export an initialized CPU reference, also used by offline tests."""
        state = module.initial_state(2)
        text = self.v41_config.text_config
        names = ["input_ids", "position_ids", "token_types", "inputs_embeds"]
        names.extend(module.state_names)
        outputs = ["logits"] + [f"present.{name}" for name in module.state_names]
        ids = torch.zeros(2, 3, dtype=torch.int64)
        positions = torch.arange(3, dtype=torch.int64).expand(2, -1)
        types = torch.full_like(ids, -1)
        dtype = module.embed.weight.dtype
        embeds = torch.zeros(2, 3, text.hidden_size, dtype=dtype)
        _, *seeded = module(ids, positions, types, embeds, *state)
        positions = positions + 3
        axes = {name: {0: "batch_size", 1: "sequence_length"} for name in names[:4] + ["logits"]}
        for name, output, value in zip(module.state_names, outputs[1:], seeded):
            dimensions = {0: "batch_size"} if value.ndim else {}
            if value.ndim > 1:
                dimensions[1] = f"{name}_length"
            axes[name] = dimensions
            axes[output] = {axis: label.replace("_length", "_present_length") for axis, label in dimensions.items()}
        self._components["decoder"] = trace_v41_model(module, (ids, positions, types, embeds, *seeded), names, outputs, axes)
        self._manifest = {
            "format": "deepseek_v41",
            "version": 1,
            "decoder": self.filename,
            "genai_supported": False,
            "genai_issue": "https://github.com/xadupre/mbext/issues/604",
            "hidden_size": text.hidden_size,
            "vocab_size": text.vocab_size,
            "image_token_id": self.v41_config.image_token_id,
            "eos_token_id": self.v41_config.eos_token_id,
            "context_length": text.max_position_embeddings,
            "position_state": "layer_0_window_positions",
            "io_dtype": str(dtype).removeprefix("torch."),
            "state": [
                {
                    "input": name,
                    "output": f"present.{name}",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                    "initial": _json_state((value[:1] if value.ndim else value).tolist()),
                }
                for name, value in zip(module.state_names, state)
            ],
        }
        vision = self.v41_config.vision_config
        if vision is not None and vision.num_hidden_layers:
            patch_size = vision.patch_size
            patches = torch.zeros(6, 3, patch_size, patch_size, dtype=dtype)
            grid = torch.tensor([2, 3], dtype=torch.int64)
            self._components["vision"] = trace_v41_model(
                _VisionGraph(module),
                (patches, grid),
                ["patches", "grid_shape"],
                ["image_features", "image_start", "image_end", "image_newline"],
                {"patches": {0: "num_patches"}, "image_features": {0: "num_image_tokens"}},
            )
            self._manifest["vision"] = os.path.splitext(self.filename)[0] + ".vision.onnx"
            self._manifest["vision_config"] = vision.to_dict()

    def save_model(self, out_dir):
        filename = self.filename
        try:
            super().save_model(out_dir)
            if "vision" in self._components:
                self._active_component = "vision"
                self.filename = self._manifest["vision"]
                super().save_model(out_dir)
        finally:
            self.filename = filename
            self._active_component = "decoder"

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        if self._manifest is None:
            raise ValueError("DeepSeek-V4.1 config-only export needs explicit state metadata.")
        path = os.path.join(out_dir, "deepseek_v41_config.json")
        with open(path, "w") as stream:
            json.dump(self._manifest, stream, indent=2)
        print("DeepSeek-V4.1 requires DeepseekV41Session for recurrent state; " "native GenAI support is tracked in #604.")
        return self._manifest
