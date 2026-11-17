# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Execution of DeepSeek-V4.1 graphs with explicit heterogeneous state."""

import json
import os

import ml_dtypes
import numpy as np


class DeepseekV41Session:
    """Run a V4.1 export using ONNX Runtime or either onnx-light backend.

    Unlike a conventional decoder, V4.1 carries integer n-gram history and
    compressor state in addition to shared attention caches. All state remains
    explicit, so independent sequences can safely share a session.
    """

    def __init__(self, directory, *, backend="onnxruntime", providers=None):
        self.directory = os.fspath(directory)
        with open(os.path.join(directory, "deepseek_v41_config.json")) as stream:
            self.config = json.load(stream)
        if self.config.get("format") != "deepseek_v41" or self.config.get("version") != 1:
            raise ValueError("Unsupported DeepSeek-V4.1 export manifest.")
        if backend not in {"onnxruntime", "onnx-light", "onnx-light-cpu"}:
            raise ValueError(f"Unknown DeepSeek-V4.1 runtime backend {backend!r}.")
        self.backend = backend
        self.providers = providers
        self.decoder = self._session(self.config["decoder"])
        self.vision = None

    def _session(self, filename):
        path = os.path.abspath(os.path.join(self.directory, filename))
        if os.path.commonpath((os.path.abspath(self.directory), path)) != os.path.abspath(self.directory):
            raise ValueError("Component path escapes the DeepSeek export directory.")
        if self.backend == "onnxruntime":
            from onnxruntime import InferenceSession

            return InferenceSession(path, providers=self.providers or ["CPUExecutionProvider"])
        from onnx_light.onnx import load_model
        from onnx_light.onnx.reference import ReferenceEvaluator

        session = ReferenceEvaluator(load_model(path, load_external_data=True))
        if self.backend == "onnx-light-cpu":
            from onnx_light_cpu import register_kernels_for_session

            register_kernels_for_session(session)
        return session

    def _run(self, session, names, feeds):
        if self.backend != "onnxruntime" or self.config["io_dtype"] != "bfloat16":
            return session.run(names, feeds)
        import torch
        from onnx_light.onnx import TensorProto
        from onnxruntime import OrtValue

        from .onnx_helper import torch_tensor_to_numpy

        # ORT's NumPy bridge does not accept BF16. Bind its original bits and
        # let ORT allocate dynamic outputs, including initially empty caches.
        buffers = {name: np.ascontiguousarray(value).reshape(value.shape) for name, value in feeds.items()}
        binding = session.io_binding()
        for name, value in buffers.items():
            if value.dtype == ml_dtypes.bfloat16:
                binding.bind_ortvalue_input(name, OrtValue.ortvalue_from_numpy_with_onnx_type(value.view(np.uint16), TensorProto.BFLOAT16))
            else:
                binding.bind_cpu_input(name, value)
        for name in names:
            binding.bind_output(name, "cpu")
        session.run_with_iobinding(binding)
        return [
            torch_tensor_to_numpy(torch.from_dlpack(value)) if value.element_type() == TensorProto.BFLOAT16 else value.numpy()
            for value in binding.get_outputs()
        ]

    @staticmethod
    def _dtype(name):
        return ml_dtypes.bfloat16 if name == "bfloat16" else np.dtype(name)

    def initial_state(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        state = {}
        for spec in self.config["state"]:
            dtype = self._dtype(spec["dtype"])
            shape = list(spec["shape"])
            if shape:
                shape[0] = batch_size
            if "initial" in spec:
                initial = np.asarray(spec["initial"], dtype=dtype)
                initial = initial.reshape([1, *shape[1:]] if shape else [])
                state[spec["input"]] = np.broadcast_to(initial, shape).copy() if shape else initial.copy()
            else:
                raise ValueError(f"Missing initial state for {spec['input']!r}.")
        return state

    def encode_image(self, image):
        """Return image token types and aligned embeddings for one local image."""
        from .deepseek_v41_image import DeepseekV41ImageProcessor

        if "vision" not in self.config:
            raise ValueError("This DeepSeek-V4.1 export has no vision component.")
        if self.vision is None:
            self.vision = self._session(self.config["vision"])
        processor = DeepseekV41ImageProcessor(self.config["vision_config"], dtype=self._dtype(self.config["io_dtype"]))
        patches, grid, types = processor(image)
        features, start, end, newline = self._run(
            self.vision, ["image_features", "image_start", "image_end", "image_newline"], {"patches": patches, "grid_shape": grid}
        )
        embeddings = np.empty((len(types), self.config["hidden_size"]), dtype=features.dtype)
        embeddings[types == 0] = start
        embeddings[types == 1] = features
        embeddings[types == 2] = newline
        embeddings[types == 3] = end
        return types, embeddings

    def run(self, input_ids, *, position_ids, token_types=None, inputs_embeds=None, state=None):
        """Return full logits and a new state dictionary without mutating inputs."""
        input_ids = np.asarray(input_ids, dtype=np.int64)
        positions = np.asarray(position_ids, dtype=np.int64)
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("input_ids must have nonempty [batch, sequence] shape.")
        if positions.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids.")
        start = int(positions[0, 0])
        contiguous = np.arange(start, start + input_ids.shape[1])
        if start < 0 or not np.all(positions == contiguous):
            raise ValueError("position_ids must be contiguous and synchronized across the batch; " "padding is not supported.")
        if start + input_ids.shape[1] > self.config["context_length"]:
            raise ValueError("position_ids exceed the model context length.")
        if np.any(input_ids < 0) or np.any(input_ids >= self.config["vocab_size"]):
            raise ValueError("input_ids contains a token outside the model vocabulary.")
        if token_types is None:
            token_types = np.full(input_ids.shape, -1, dtype=np.int64)
        token_types = np.asarray(token_types, dtype=np.int64)
        if token_types.shape != input_ids.shape:
            raise ValueError("token_types must have the same shape as input_ids.")
        if np.any((token_types < -1) | (token_types > 3)):
            raise ValueError("token_types must be -1 (text) or 0 through 3 (image).")
        if np.any(token_types >= 0) and "vision" not in self.config:
            raise ValueError("This DeepSeek-V4.1 export has no vision component.")
        if inputs_embeds is None:
            if np.any(token_types >= 0):
                raise ValueError("Image positions require precomputed image embeddings.")
            inputs_embeds = np.zeros((*input_ids.shape, self.config["hidden_size"]), dtype=self._dtype(self.config["io_dtype"]))
        if inputs_embeds.shape != (*input_ids.shape, self.config["hidden_size"]):
            raise ValueError("inputs_embeds has an incompatible shape.")
        if state is None:
            if start != 0:
                raise ValueError("A new sequence must start at position zero.")
            state = self.initial_state(input_ids.shape[0])
        expected = {spec["input"] for spec in self.config["state"]}
        if set(state) != expected:
            raise ValueError("Recurrent state names do not match the export manifest.")
        position_state = self.config.get("position_state")
        if position_state is not None:
            previous = np.asarray(state[position_state])
            if (
                previous.ndim != 2
                or previous.shape[0] != input_ids.shape[0]
                or previous.shape[1] == 0
                or not np.all(previous[:, -1] + 1 == start)
            ):
                raise ValueError("position_ids do not continue the recurrent state.")
        feeds = {
            "input_ids": input_ids,
            "position_ids": positions,
            "token_types": token_types,
            "inputs_embeds": np.asarray(inputs_embeds, dtype=self._dtype(self.config["io_dtype"])),
            **state,
        }
        names = ["logits"] + [spec["output"] for spec in self.config["state"]]
        result = self._run(self.decoder, names, feeds)
        return result[0], {spec["input"]: value for spec, value in zip(self.config["state"], result[1:])}

    def generate(self, input_ids, *, max_new_tokens=10, token_types=None, inputs_embeds=None, eos_token_id=None):
        """Greedy generation with bounded local state and compressed shared caches."""
        tokens = np.asarray(input_ids, dtype=np.int64)
        if tokens.ndim != 2 or not all(tokens.shape):
            raise ValueError("input_ids must have nonempty [batch, sequence] shape.")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer.")
        if tokens.shape[1] + max_new_tokens > self.config["context_length"]:
            raise ValueError("Requested generation exceeds the model context length.")
        if max_new_tokens == 0:
            return tokens.copy()
        state = self.initial_state(tokens.shape[0])
        positions = np.broadcast_to(np.arange(tokens.shape[1]), tokens.shape)
        logits, state = self.run(tokens, position_ids=positions, token_types=token_types, inputs_embeds=inputs_embeds, state=state)
        chunks = [tokens]
        eos = self.config["eos_token_id"] if eos_token_id is None else eos_token_id
        eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
        finished = np.zeros(tokens.shape[0], dtype=bool)
        for step in range(max_new_tokens):
            next_tokens = logits[:, -1].argmax(-1).astype(np.int64)
            if eos:
                next_tokens = np.where(finished, eos[0], next_tokens)
                finished |= np.isin(next_tokens, eos)
            chunks.append(next_tokens[:, None])
            if finished.all() or step + 1 == max_new_tokens:
                break
            positions = np.full((tokens.shape[0], 1), tokens.shape[1] + step, dtype=np.int64)
            logits, state = self.run(next_tokens[:, None], position_ids=positions, state=state)
        return np.concatenate(chunks, axis=1)
