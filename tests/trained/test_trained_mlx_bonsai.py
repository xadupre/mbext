# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Opt-in greedy-token parity check against the public MLX checkpoint."""

import os
import sys

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, long_test


class TestTrainedMLXBonsai(ExtTestCase):
    @long_test()
    def test_first_generated_token(self):
        if sys.platform != "darwin":
            self.skipTest("MLX requires an Apple Silicon host.")

        import mlx_lm
        import onnxruntime_genai as og
        from huggingface_hub import snapshot_download

        from modelbuilder.builder import create_model

        model_id = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
        output, cache = self.get_dirs("test_trained_mlx_bonsai", clean=False)
        output = os.path.abspath(output)
        source = snapshot_download(repo_id=model_id, cache_dir=cache)
        create_model(
            model_name=model_id,
            input_path=source,
            output_dir=output,
            precision="int2",
            execution_provider="cpu",
            cache_dir=cache,
            mlx_quantized=True,
        )
        mlx_model, tokenizer = mlx_lm.load(source)
        prompt = "The capital of France is"
        tokens = tokenizer.encode(prompt)
        mlx_token = next(mlx_lm.stream_generate(mlx_model, tokenizer, tokens, max_tokens=1)).token

        ort_model = og.Model(output)
        params = og.GeneratorParams(ort_model)
        params.set_search_options(do_sample=False, max_length=len(tokens) + 1, top_k=1)
        generator = og.Generator(ort_model, params)
        generator.append_tokens(np.array(tokens, dtype=np.int32))
        generator.generate_next_token()
        self.assertEqual(int(generator.get_next_tokens()[0]), int(mlx_token))
