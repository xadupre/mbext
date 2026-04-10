# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Helpers for running ONNX models with :class:`onnxruntime.InferenceSession`.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import onnx

# ---------------------------------------------------------------------------
# OnnxRuntime type-string → numpy dtype mapping
# ---------------------------------------------------------------------------

_ORT_TYPE_TO_NUMPY: Dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float32)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(float64)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _ort_type_to_numpy_dtype(ort_type: str) -> type:
    """Converts an OnnxRuntime type string (e.g. ``"tensor(float)"``) to a NumPy dtype.

    :param ort_type: type string returned by ``NodeArg.type``
    :return: corresponding NumPy dtype
    :raises ValueError: when the type is unknown
    """
    try:
        return _ORT_TYPE_TO_NUMPY[ort_type]
    except KeyError:
        if "bfloat16" in ort_type:
            try:
                import ml_dtypes  # type: ignore[import]

                return ml_dtypes.bfloat16
            except ImportError:
                pass
        raise ValueError(
            f"Unknown OnnxRuntime type string {ort_type!r}. " f"Known types: {sorted(_ORT_TYPE_TO_NUMPY)}"
        ) from None


def _get_dim(i: int, s: Optional[Union[str, int]], batch: int = 1) -> int:
    """Returns a concrete integer dimension from a symbolic or integer shape element.

    :param i: position of the dimension (0 = batch)
    :param s: dimension value (``int``, ``str``, or ``None`` for dynamic dims)
    :param batch: batch size to use for the batch dimension
    :return: concrete integer dimension
    """
    if isinstance(s, int):
        return s
    # None or string symbolic dim
    if i == 0:
        return batch
    # Everything else (cache length, sequence length) starts empty.
    return 0


# Inputs that are never treated as KV-cache slots.
_KNOWN_NON_CACHE: FrozenSet[str] = frozenset(
    {"input_ids", "attention_mask", "position_ids", "token_type_ids", "cache_position"}
)


def _make_empty_cache(
    batch: int,
    cache_names: List[str],
    cache_shapes: List[Tuple],
    cache_types: List[str],
) -> Dict[str, np.ndarray]:
    """Creates zero-filled KV-cache arrays for the first generation step.

    :param batch: batch size
    :param cache_names: names of the KV-cache inputs
    :param cache_shapes: ORT input shapes for those inputs
    :param cache_types: ORT type strings for those inputs
    :return: dict ``{name: zero ndarray}``
    """
    feeds: Dict[str, np.ndarray] = {}
    for name, shape, ort_type in zip(cache_names, cache_shapes, cache_types):
        new_shape = tuple(_get_dim(i, s, batch=batch) for i, s in enumerate(shape))
        if not new_shape or new_shape[0] <= 0:
            raise ValueError(f"new_shape={new_shape} cannot have a null batch size, " f"name={name!r}, shape={shape}")
        dtype = _ort_type_to_numpy_dtype(ort_type)
        feeds[name] = np.zeros(new_shape, dtype=dtype)
    return feeds


def onnx_generate(
    model_or_session: Union[str, onnx.ModelProto, "onnxruntime.InferenceSession"],  # noqa: F821
    input_ids: np.ndarray,
    attention_mask: Optional[np.ndarray] = None,
    eos_token_id: Optional[int] = None,
    max_new_tokens: int = 20,
    do_sample: bool = False,
    return_session: bool = False,
    verbose: int = 0,
) -> Union[np.ndarray, Tuple]:
    """
    Performs auto-regressive token generation using an exported ONNX model
    and :class:`onnxruntime.InferenceSession`.

    The function mimics the ``generate`` method of HuggingFace *transformers*
    models.  It calls the ONNX forward pass in a loop, appending the most
    likely next token at each step (greedy decoding by default), and feeds the
    updated *past key/value* tensors back into the model on each subsequent
    call.

    Models that do **not** expose past-key-value inputs/outputs are also
    supported: in that case the full ``input_ids`` sequence is fed on every
    step (simpler but less efficient).

    :param model_or_session: path to an ``.onnx`` file, a
        :class:`onnx.ModelProto` loaded into memory, or an already-created
        :class:`onnxruntime.InferenceSession`.
    :param input_ids: initial prompt token IDs, integer array of shape
        ``[batch, seq_len]``.
    :param attention_mask: optional attention mask of shape
        ``[batch, seq_len]``.  When *None*, an all-ones mask matching
        ``input_ids`` is created automatically.
    :param eos_token_id: when set, generation stops as soon as *all* batch
        items have produced this token.
    :param max_new_tokens: upper bound on the number of tokens to generate
        (not counting the original ``input_ids``).
    :param do_sample: when *True* sample the next token from the softmax
        distribution; when *False* (default) use greedy argmax.
    :param return_session: when *True* return a 3-tuple
        ``(tokens, session, last_feeds)`` instead of just the tokens.
    :param verbose: verbosity level (0 = silent).
    :return: integer array of shape ``[batch, seq_len + generated_tokens]``
        containing the original prompt followed by the generated tokens.
        When ``return_session=True``, returns a 3-tuple
        ``(tokens, session, last_feeds)``.

    Example with a tiny synthetic ONNX decoder (no KV cache)::

        import numpy as np
        import onnx
        import onnx.helper as oh
        import onnx.numpy_helper as onh
        from modelbuilder.helpers.rt_helper import onnx_generate

        TINT64 = onnx.TensorProto.INT64
        TFLOAT = onnx.TensorProto.FLOAT
        VOCAB  = 8

        # A minimal "LM head": always returns the same logits so that the
        # argmax always picks token 3.
        fixed_logits = np.zeros((1, 1, VOCAB), dtype=np.float32)
        fixed_logits[0, 0, 3] = 10.0   # token 3 always wins

        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node(
                        "Constant",
                        [],
                        ["logits"],
                        value=onh.from_array(fixed_logits),
                    ),
                ],
                "tiny_lm",
                [oh.make_tensor_value_info("input_ids", TINT64, [1, None])],
                [oh.make_tensor_value_info("logits", TFLOAT, [1, 1, VOCAB])],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=9,
        )

        prompt = np.array([[1, 2]], dtype=np.int64)
        tokens = onnx_generate(model, prompt, max_new_tokens=3, eos_token_id=3)
        # tokens == [[1, 2, 3]]  (stops after the first EOS token)

    .. note::
        When the ONNX model exposes *past key/value* inputs, the function
        automatically creates zero-filled tensors for the initial call and
        feeds back the corresponding outputs on every subsequent step.  The
        KV-cache heuristic treats any input whose name is **not** in
        ``{input_ids, attention_mask, position_ids, token_type_ids,
        cache_position}`` as a KV-cache slot.  Present-key/value outputs are
        mapped back to past-key/value inputs by position (i.e. ``outputs[1]``
        → ``cache_inputs[0]``, etc.).
    """
    from onnxruntime import InferenceSession

    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be a 2-D array [batch, seq_len], got shape {input_ids.shape}")
    input_ids = np.asarray(input_ids, dtype=np.int64)

    if not isinstance(model_or_session, InferenceSession):
        providers = ["CPUExecutionProvider"]
        if isinstance(model_or_session, onnx.ModelProto):
            session_input = model_or_session.SerializeToString()
        else:
            session_input = model_or_session
        session = InferenceSession(session_input, providers=providers)
    else:
        session = model_or_session

    input_meta = session.get_inputs()
    input_names: List[str] = [m.name for m in input_meta]
    input_shapes = [m.shape for m in input_meta]
    input_types: List[str] = [m.type for m in input_meta]

    has_position_ids = "position_ids" in input_names
    has_cache_position = "cache_position" in input_names
    has_attention_mask = "attention_mask" in input_names

    batch_size = input_ids.shape[0]

    # Heuristic KV-cache detection: any input not in the set of known
    # "meta" inputs is treated as a past-key/value slot.
    cache_names = [n for n in input_names if n not in _KNOWN_NON_CACHE]
    cache_shapes = [input_shapes[input_names.index(n)] for n in cache_names]
    cache_types = [input_types[input_names.index(n)] for n in cache_names]

    # Build the initial attention mask (all ones if not provided).
    if attention_mask is None:
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    else:
        attention_mask = np.asarray(attention_mask, dtype=np.int64)

    # Bootstrap zero-filled KV-cache arrays.
    empty_cache = _make_empty_cache(batch_size, cache_names, cache_shapes, cache_types)

    # ------------------------------------------------------------------ #
    # Prefill step                                                         #
    # ------------------------------------------------------------------ #
    feeds: Dict[str, np.ndarray] = {"input_ids": input_ids}
    if has_attention_mask:
        feeds["attention_mask"] = attention_mask
    feeds.update(empty_cache)

    if has_position_ids:
        seq_len = input_ids.shape[1]
        if seq_len <= 0:
            raise ValueError(f"unexpected value for input_ids shape={input_ids.shape}")
        feeds["position_ids"] = np.tile(np.arange(seq_len, dtype=np.int64), (batch_size, 1))

    if has_cache_position:
        past_len = (
            next(iter(empty_cache.values())).shape[2]
            if empty_cache and next(iter(empty_cache.values())).ndim > 2
            else 0
        )
        feeds["cache_position"] = np.arange(past_len, input_ids.shape[1] + past_len, dtype=np.int64)

    if verbose:
        print(f"[onnx_generate] prefill feeds: {list(feeds)}")

    outputs = session.run(None, feeds)

    # ------------------------------------------------------------------ #
    # Decode loop                                                          #
    # ------------------------------------------------------------------ #
    last_position = 0
    # Per-batch EOS tracking so that the loop terminates only when *all*
    # sequences have finished.
    eos_found = np.zeros(batch_size, dtype=np.bool_)

    for step in range(max_new_tokens):
        next_token_logits = outputs[0][:, -1, :]  # [batch, vocab]

        if do_sample:
            # Sample from the probability distribution over the vocabulary.
            def _softmax(x: np.ndarray) -> np.ndarray:
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e / e.sum(axis=-1, keepdims=True)

            probs = _softmax(next_token_logits)
            next_token_id = np.array(
                [np.random.choice(probs.shape[-1], p=probs[b]) for b in range(batch_size)],
                dtype=np.int64,
            ).reshape(batch_size, 1)
        else:
            # Greedy decoding: take the argmax token.
            next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True).astype(np.int64)  # [batch, 1]

        # Update per-batch EOS flags.
        if eos_token_id is not None:
            eos_found |= next_token_id[:, 0] == eos_token_id

        input_ids = np.concatenate([input_ids, next_token_id], axis=-1)

        if verbose:
            print(f"[onnx_generate] step {step}: next_token_id={next_token_id.tolist()}")

        # Stop once every sequence in the batch has produced EOS.
        if eos_token_id is not None and eos_found.all():
            break

        # Extend the attention mask by one column of ones for the new token.
        attention_mask = np.concatenate(
            [attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)],
            axis=-1,
        )

        # Build feeds for the next decode step.
        if not cache_names:
            # No KV cache: feed the full growing sequence.
            feeds = {"input_ids": input_ids}
            if has_attention_mask:
                feeds["attention_mask"] = attention_mask
        else:
            # KV cache: feed only the single new token; map present outputs
            # back to past inputs by position.
            feeds = {"input_ids": next_token_id}
            if has_attention_mask:
                feeds["attention_mask"] = attention_mask
            for j, name in enumerate(cache_names):
                if 1 + j < len(outputs):
                    feeds[name] = outputs[1 + j]

        if has_position_ids or has_cache_position:
            last_position = input_ids.shape[1] - 1

        if has_position_ids:
            feeds["position_ids"] = np.full((batch_size, 1), last_position, dtype=np.int64)

        if has_cache_position:
            feeds["cache_position"] = np.arange(last_position, last_position + 1, dtype=np.int64)

        outputs = session.run(None, feeds)

    if return_session:
        return input_ids, session, feeds
    return input_ids
