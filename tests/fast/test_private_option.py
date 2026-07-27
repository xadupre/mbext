# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import tempfile
import textwrap
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

# A private model reusing the Qwen3 backbone but exposed under a custom
# architecture name that is not part of the built-in dispatch in
# modelbuilder.builder.create_model.
PRIVATE_ARCHITECTURE = "PrivateQwenForCausalLM"
PRIVATE_MODEL_NAME = "private/PrivateQwen"

MODELING_FILE_CONTENT = textwrap.dedent("""\
    # Imported before the config is loaded. A real modeling file would register
    # its custom architecture with transformers here.
    PRIVATE_MODELING_IMPORTED = True
    """)

CONVERT_FILE_CONTENT = textwrap.dedent("""\
    from modelbuilder.builders.qwen import Qwen3Model


    class PrivateQwenModel(Qwen3Model):
        pass
    """)

CONVERT_FILE_WITH_BUILDER_ATTR = textwrap.dedent("""\
    from modelbuilder.builders.qwen import Qwen3Model


    class PrivateQwenModelA(Qwen3Model):
        pass


    class PrivateQwenModelB(Qwen3Model):
        pass


    MODEL_BUILDER = PrivateQwenModelA
    """)


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestParsePrivateOption(ExtTestCase):
    """Unit tests for the --private option parsing/loading helpers."""

    def test_parse_private_option_full(self):
        from modelbuilder.builder import parse_private_option

        self.assertEqual(parse_private_option("m.py;c.py;t.py"), ("m.py", "c.py", "t.py"))

    def test_parse_private_option_empty_modeling(self):
        from modelbuilder.builder import parse_private_option

        self.assertEqual(parse_private_option(";c.py;t.py"), ("", "c.py", "t.py"))

    def test_parse_private_option_only_convert(self):
        from modelbuilder.builder import parse_private_option

        self.assertEqual(parse_private_option("c.py"), ("c.py", "", ""))

    def test_parse_private_option_strips_whitespace(self):
        from modelbuilder.builder import parse_private_option

        self.assertEqual(parse_private_option(" m.py ; c.py ; t.py "), ("m.py", "c.py", "t.py"))

    def test_load_private_model_builder_single_subclass(self):
        from modelbuilder.builder import load_private_model_builder

        with tempfile.TemporaryDirectory() as tmp:
            convert = _write(os.path.join(tmp, "convert.py"), CONVERT_FILE_CONTENT)
            builder = load_private_model_builder(f";{convert};")
            self.assertEqual(builder.__name__, "PrivateQwenModel")

    def test_load_private_model_builder_imports_modeling_file(self):
        from modelbuilder.builder import load_private_model_builder

        with tempfile.TemporaryDirectory() as tmp:
            modeling = _write(os.path.join(tmp, "modeling.py"), MODELING_FILE_CONTENT)
            convert = _write(os.path.join(tmp, "convert.py"), CONVERT_FILE_CONTENT)
            builder = load_private_model_builder(f"{modeling};{convert};test.py")
            self.assertEqual(builder.__name__, "PrivateQwenModel")

    def test_load_private_model_builder_uses_model_builder_attr(self):
        from modelbuilder.builder import load_private_model_builder

        with tempfile.TemporaryDirectory() as tmp:
            convert = _write(os.path.join(tmp, "convert.py"), CONVERT_FILE_WITH_BUILDER_ATTR)
            builder = load_private_model_builder(f";{convert};")
            self.assertEqual(builder.__name__, "PrivateQwenModelA")

    def test_load_private_model_builder_missing_convert(self):
        from modelbuilder.builder import load_private_model_builder

        self.assertRaise(lambda: load_private_model_builder(";;"), ValueError)

    def test_load_private_model_builder_missing_file(self):
        from modelbuilder.builder import load_private_model_builder

        self.assertRaise(lambda: load_private_model_builder(";does_not_exist.py;"), FileNotFoundError)

    def test_load_private_model_builder_no_subclass(self):
        from modelbuilder.builder import load_private_model_builder

        with tempfile.TemporaryDirectory() as tmp:
            convert = _write(os.path.join(tmp, "convert.py"), "X = 1\n")
            self.assertRaise(lambda: load_private_model_builder(f";{convert};"), ValueError)

    def test_run_private_tests_executes_file(self):
        from modelbuilder.builder import run_private_tests

        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "ran.txt")
            test_file = _write(os.path.join(tmp, "test.py"), f"open({marker!r}, 'w', encoding='utf-8').write('ran')\n")
            run_private_tests(f";c.py;{test_file}")
            self.assertExists(marker)

    def test_run_private_tests_no_private(self):
        from modelbuilder.builder import run_private_tests

        self.assertRaise(lambda: run_private_tests(None), ValueError)

    def test_run_private_tests_missing_test_file(self):
        from modelbuilder.builder import run_private_tests

        self.assertRaise(lambda: run_private_tests(";c.py;"), ValueError)


class TestRandomPrivateQwen(ExtTestCase):
    """End-to-end conversion test exercising the --private option."""

    def _make_private_files(self, tmp):
        modeling = _write(os.path.join(tmp, "modeling.py"), MODELING_FILE_CONTENT)
        convert = _write(os.path.join(tmp, "convert.py"), CONVERT_FILE_CONTENT)
        test_file = _write(os.path.join(tmp, "test.py"), "# fast test placeholder\n")
        return f"{modeling};{convert};{test_file}"

    def common_fast_private_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM, Qwen3Config

        num_hidden_layers = 1

        config = Qwen3Config(
            architectures=[PRIVATE_ARCHITECTURE],
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            head_dim=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            vocab_size=32000,
            use_sliding_window=False,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        with tempfile.TemporaryDirectory() as tmp:
            private = self._make_private_files(tmp)
            self.run_random_weights_test(
                model=model,
                tokenizer=tokenizer,
                model_name=PRIVATE_MODEL_NAME,
                basename=f"test_discrepancies_private_{precision}_{provider}",
                precision=precision,
                provider=provider,
                num_hidden_layers=num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                head_size=config.head_dim,
                vocab_size=config.vocab_size,
                create_model_kwargs={"num_hidden_layers": num_hidden_layers, "private": private},
            )

    @hide_stdout()
    def test_fast_discrepancy_private_fp32_cpu(self):
        self.common_fast_private_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_private_fp16_cpu(self):
        self.common_fast_private_random_weights("fp16", "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
