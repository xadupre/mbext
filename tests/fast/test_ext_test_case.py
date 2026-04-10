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

from modelbuilder.ext_test_case import ExtTestCase, _make_json_serializable, _read_results, results_to_dataframe


class TestResultsToMarkdown(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(results_to_dataframe([]).to_markdown(), "")

    def test_single_row(self):
        rows = [{"model_id": "m", "experiment": "e", "max_abs_err": 0.001}]
        md = results_to_dataframe(rows).to_markdown()
        lines = md.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("model_id", lines[0])
        self.assertIn("experiment", lines[0])
        self.assertIn("max_abs_err", lines[0])
        # separator row
        self.assertIn("---", lines[1])
        # data row
        self.assertIn("m", lines[2])
        self.assertIn("e", lines[2])
        self.assertIn("0.001", lines[2])

    def test_multiple_rows(self):
        rows = [
            {"model_id": "model-a", "experiment": "prefill", "max_abs_err": 1e-4},
            {"model_id": "model-a", "experiment": "decode", "max_abs_err": 2e-4},
        ]
        md = results_to_dataframe(rows).to_markdown()
        lines = md.splitlines()
        # header + separator + 2 data rows
        self.assertEqual(len(lines), 4)
        self.assertIn("model-a", lines[2])
        self.assertIn("prefill", lines[2])
        self.assertIn("model-a", lines[3])
        self.assertIn("decode", lines[3])

    def test_union_of_keys(self):
        """Columns should be the union of all keys across all rows."""
        rows = [{"a": 1, "b": 2, "provider": "cpu"}, {"b": 3, "c": 4, "provider": "cpu"}]
        md = results_to_dataframe(rows).to_markdown()
        header = md.splitlines()[0]
        self.assertIn("a", header)
        self.assertIn("b", header)
        self.assertIn("c", header)

    def test_missing_value_rendered_as_empty(self):
        rows = [{"a": 1, "b": 2, "provider": "cpu"}, {"a": 3, "provider": "cpu"}]
        md = results_to_dataframe(rows).to_markdown()
        last_row = md.splitlines()[-1]
        # The second row has no 'b' key – its cell should be empty.
        self.assertIn("| cpu", last_row)

    def test_float_formatting(self):
        """Floats must be formatted with %g (no trailing zeros)."""
        rows = [{"v": 0.00100, "provider": "cpu"}]
        md = results_to_dataframe(rows).to_markdown()
        self.assertIn("0.001", md)
        self.assertNotIn("0.00100", md)

    def test_pipe_separated_columns(self):
        rows = [{"x": 1, "y": 2, "provider": "cpu"}]
        md = results_to_dataframe(rows).to_markdown()
        for line in md.splitlines():
            self.assertTrue(line.startswith("|"))
            self.assertTrue(line.endswith("|"))


class TestMakeJsonSerializable(unittest.TestCase):
    def test_numpy_dtype(self):
        self.assertEqual(_make_json_serializable(np.dtype("float32")), "float32")

    def test_numpy_integer(self):
        result = _make_json_serializable(np.int64(7))
        self.assertIsInstance(result, int)
        self.assertEqual(result, 7)

    def test_numpy_floating(self):
        result = _make_json_serializable(np.float32(1.5))
        self.assertIsInstance(result, float)

    def test_numpy_array(self):
        result = _make_json_serializable(np.array([1, 2, 3]))
        self.assertEqual(result, [1, 2, 3])

    def test_tuple_to_list(self):
        self.assertEqual(_make_json_serializable((1, 2, 3)), [1, 2, 3])

    def test_passthrough(self):
        self.assertEqual(_make_json_serializable("hello"), "hello")
        self.assertEqual(_make_json_serializable(42), 42)


class TestLogResults(ExtTestCase):
    def test_log_results_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                data = {
                    "model_id": "test-model",
                    "experiment": "prefill",
                    "max_abs_err": 1e-4,
                    "shape": (1, 5, 512),
                    "dtype": np.dtype("float32"),
                }
                self.log_results(data)

                json_path = os.path.join("stats", "end2end_results.json")
                md_path = os.path.join("stats", "end2end_results.md")

                self.assertTrue(os.path.exists(json_path))
                self.assertTrue(os.path.exists(md_path))

                # JSON file should contain a parseable JSON object.
                with open(json_path) as f:
                    line = f.readline().strip()
                record = json.loads(line)
                self.assertEqual(record["model_id"], "test-model")
                self.assertEqual(record["dtype"], "float32")
                self.assertEqual(record["shape"], [1, 5, 512])

                # Markdown file should contain a table.
                with open(md_path) as f:
                    content = f.read()
                self.assertIn("model_id", content)
                self.assertIn("test-model", content)
                self.assertIn("---", content)
            finally:
                os.chdir(orig_dir)

    def test_log_results_accumulates_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                for experiment in ("prefill", "decode"):
                    self.log_results({"model_id": "m", "experiment": experiment, "max_abs_err": 0.0})

                md_path = os.path.join("stats", "end2end_results.md")
                with open(md_path) as f:
                    content = f.read()
                # header + separator + 2 data rows = 4 lines
                lines = [ln for ln in content.splitlines() if ln.strip()]
                self.assertEqual(len(lines), 4)
                self.assertIn("prefill", content)
                self.assertIn("decode", content)
            finally:
                os.chdir(orig_dir)


class TestReadResults(unittest.TestCase):
    def test_read_json_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write(json.dumps({"a": 2}) + "\n")
            fname = f.name
        try:
            results = _read_results(fname)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["a"], 1)
            self.assertEqual(results[1]["a"], 2)
        finally:
            os.unlink(fname)

    def test_skip_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write("\n")
            f.write(json.dumps({"a": 2}) + "\n")
            fname = f.name
        try:
            results = _read_results(fname)
            self.assertEqual(len(results), 2)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main(verbosity=2)
