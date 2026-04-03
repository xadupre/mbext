import contextlib
import io
import os
import shutil
import unittest
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


def hide_stdout(f: Optional[Callable] = None) -> Callable:
    """
    Catches warnings, hides standard output.
    The function may be disabled by setting ``UNHIDE=1``
    before running the unit test.

    :param f: the function is called with the stdout as an argument
    """

    def wrapper(fct):
        def call_f(self):
            if os.environ.get("UNHIDE", "") in (1, "1", "True", "true"):
                fct(self)
                return
            st = io.StringIO()
            with contextlib.redirect_stdout(st), warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))
                try:
                    fct(self)
                except AssertionError as e:  # pragma: no cover
                    if "torch is not recent enough, file" in str(e):
                        raise unittest.SkipTest(str(e))  # noqa: B904
                    raise
            if f is not None:
                f(st.getvalue())
            return None

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def _msg(msg: Union[Callable[[], str], str], add_bracket: bool = True) -> str:
    if add_bracket:
        m = _msg(msg, add_bracket=False)
        if m:
            if "\n" in m:
                return f"\n----\n{m}\n---\n"
            return f" ({m})"
        return ""
    if callable(msg):
        return msg()
    return msg or ""


def long_test(msg: Optional[Union[Callable[[], str], str]] = None) -> Callable:
    """Skips a unit test if it runs on :epkg:`azure pipeline` on :epkg:`Windows`."""
    if os.environ.get("LONGTEST", "0") in ("0", 0, False, "False", "false"):
        msg = f"Skipped (set LONGTEST=1 to run it. {_msg(msg)}"
        return unittest.skip(msg)
    return lambda x: x


class ExtTestCase(unittest.TestCase):
    _warns = []

    def shortDescription(self):
        # To remove annoying display on the screen every time verbosity is enabled.
        return None

    def clean_dir(self, path: str):
        """Removes a directory and all its contents if it exists."""
        if os.path.exists(path):
            shutil.rmtree(path)

    def get_dirs(self, prefix: str) -> Tuple[str]:
        cache_dir = f"dump_models/{prefix}/cache"
        output_dir = f"dump_models/{prefix}/output"
        cache_dir = f"dump_models/{prefix}/cache"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        self.addCleanup(self.clean_dir, f"dump_models/{prefix}")
        return output_dir, cache_dir

    def get_model_dir(self, prefix: str) -> Tuple[str]:
        model_dir = f"dump_models/{prefix}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        self.addCleanup(self.clean_dir, f"dump_models/{prefix}")
        return model_dir

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertEqualArray(
        self,
        expected: np.ndarray,
        value: np.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        np.testing.assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertAlmostEqual(
        self,
        expected: np.ndarray,
        value: np.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, np.ndarray):
            expected = np.array(expected)
        if not isinstance(value, np.ndarray):
            value = np.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(
        self, fct: Callable, exc_type: Exception, msg: Optional[str] = None
    ):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.") from e
            if msg is None:
                return
            if msg not in str(e):
                raise AssertionError(f"Unexpected error message {e!r}.") from e
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if len(value) == 0:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) == 0:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string {full!r}.")

    def check_ort(
        self, onx: Union["onnx.ModelProto", str]  # noqa: F821
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        return self._check_with_ort(onx, cpu=True)

    def _check_with_ort(
        self, proto: Union["onnx.ModelProto", str], cpu: bool = False  # noqa: F821
    ) -> "onnxruntime.InferenceSession":  # noqa: F821
        from onnxruntime import InferenceSession, get_available_providers

        providers = ["CPUExecutionProvider"]
        if not cpu and "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        return InferenceSession(
            proto.SerializeToString() if hasattr(proto, "SerializeToString") else proto,
            providers=providers,
        )
