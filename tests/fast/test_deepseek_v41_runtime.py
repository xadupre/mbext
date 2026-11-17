# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import tempfile
import unittest

import numpy as np
import torch
from onnx_light.onnx import save_model

from modelbuilder.builders.deepseek_v41 import trace_v41_model
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout
from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session


class _HistoryDecoder(torch.nn.Module):
    def forward(self, input_ids, position_ids, token_types, inputs_embeds, values, tokens):
        target = (
            input_ids + 1 + tokens.sum(1, keepdim=True) + position_ids * 0 + (token_types + 1) * 0 + inputs_embeds[:, :, 0].long() * 0
        ) % 16
        return (
            torch.nn.functional.one_hot(target, 16).float(),
            torch.cat((values, input_ids.to(values.dtype).unsqueeze(-1)), 1)[:, -2:],
            torch.cat((tokens, input_ids), 1)[:, -2:],
        )


class TestDeepseekV41Runtime(ExtTestCase):
    def export_history(self, directory, dtype=torch.float32):
        names = ["input_ids", "position_ids", "token_types", "inputs_embeds", "values", "tokens"]
        outputs = ["logits", "present.values", "present.tokens"]
        ids = torch.tensor([[1, 2]], dtype=torch.int64)
        args = (
            ids,
            ids - 1,
            torch.full_like(ids, -1),
            torch.zeros(1, 2, 1, dtype=dtype),
            torch.zeros(1, 1, 1, dtype=dtype),
            torch.zeros(1, 1, dtype=torch.int64),
        )
        axes = {name: {0: "batch", 1: name + "_length"} for name in names + outputs}
        graph = trace_v41_model(_HistoryDecoder(), args, names, outputs, axes)
        config = {
            "format": "deepseek_v41",
            "version": 1,
            "decoder": "model.onnx",
            "hidden_size": 1,
            "vocab_size": 16,
            "io_dtype": str(dtype).removeprefix("torch."),
            "eos_token_id": None,
            "context_length": 32,
            "state": [
                dict(input="values", output="present.values", shape=[1, 0, 1], dtype=str(dtype).removeprefix("torch."), initial=[[]]),
                dict(input="tokens", output="present.tokens", shape=[1, 0], dtype="int64", initial=[[]]),
            ],
        }
        save_model(graph, os.path.join(directory, "model.onnx"))
        with open(os.path.join(directory, "deepseek_v41_config.json"), "w") as stream:
            json.dump(config, stream)

    @hide_stdout()
    def test_empty_heterogeneous_state_and_cached_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            self.export_history(directory)
            for backend in ("onnxruntime", "onnx-light", "onnx-light-cpu"):
                with self.subTest(backend=backend):
                    session = DeepseekV41Session(directory, backend=backend)
                    initial = session.initial_state(2)
                    self.assertEqual(initial["values"].shape, (2, 0, 1))
                    self.assertEqual(initial["tokens"].dtype, np.dtype("int64"))
                    actual = session.generate(np.array([[1, 2]], dtype=np.int64), max_new_tokens=3)
                    np.testing.assert_array_equal(actual, [[1, 2, 3, 7, 13]])
                    stopped = session.generate(np.array([[1, 2]], dtype=np.int64), max_new_tokens=3, eos_token_id=7)
                    np.testing.assert_array_equal(stopped, [[1, 2, 3, 7]])
                    for positions, message in (
                        ([[1, 2]], "position zero"),
                        ([[0, 2]], "contiguous"),
                        ([[-1, 0]], "contiguous"),
                        ([[31, 32]], "context length"),
                    ):
                        with self.subTest(positions=positions):
                            with self.assertRaisesRegex(ValueError, message):
                                session.run([[1, 2]], position_ids=positions)
                    with self.assertRaisesRegex(ValueError, "synchronized"):
                        session.run([[1, 2], [1, 2]], position_ids=[[0, 1], [1, 2]])
                    with self.assertRaisesRegex(ValueError, "token_types"):
                        session.run([[1, 2]], position_ids=[[0, 1]], token_types=[[-1, 4]])

    @hide_stdout()
    def test_bf16_dynamic_state_binding_and_output_lifetime(self):
        import ml_dtypes

        # This graph needs only casts and byte-moving BF16 kernels, allowing
        # the real ORT binding path to be checked without CUDA arithmetic.
        with tempfile.TemporaryDirectory() as directory:
            self.export_history(directory, dtype=torch.bfloat16)
            session = DeepseekV41Session(directory)
            initial = session.initial_state(1)
            self.assertEqual(initial["values"].dtype, np.dtype(ml_dtypes.bfloat16))
            self.assertEqual(initial["values"].shape, (1, 0, 1))
            _, state = session.run([[1, 2]], position_ids=[[0, 1]], state=initial)
            self.assertEqual(state["values"].dtype, np.dtype(ml_dtypes.bfloat16))
            np.testing.assert_array_equal(state["values"].astype(np.float32), [[[1], [2]]])
            _, following = session.run([[3]], position_ids=[[2]], state=state)
            np.testing.assert_array_equal(following["values"].astype(np.float32), [[[2], [3]]])
            np.testing.assert_array_equal(state["values"].astype(np.float32), [[[1], [2]]])
            np.testing.assert_array_equal(session.generate([[1, 2]], max_new_tokens=3), [[1, 2, 3, 7, 13]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
