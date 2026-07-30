Differences with the onnxruntime-genai model builder
====================================================

mbext started as a fork of the model builder shipped with `onnxruntime-genai
<https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models>`_
(``onnxruntime_genai.models.builder``). It keeps the same command-line interface
and the same overall approach — build the ONNX graph directly rather than
tracing PyTorch — so a conversion command is almost identical:

.. code-block:: bash

    # upstream
    python -m onnxruntime_genai.models.builder -i <model> -o <out> -p int4 -e cpu

    # mbext
    python -m modelbuilder.builder -i <model> -o <out> -p int4 -e cpu

What mbext adds
---------------

**Fast, offline tests.** The headline difference is the test suite. mbext
validates every builder with tiny, randomly-initialised models that run in
seconds and never download weights, checking numerical discrepancies against the
PyTorch reference. This gives a short CI and quick, deterministic feedback (see
:doc:`design`).

**More architectures.** mbext supports additional model families on top of the
upstream set. The full, always up-to-date list is on the
:doc:`architectures` page.

**A ``--private`` option.** Custom or proprietary architectures can be converted
from files that live outside the package, without patching the dispatch chain
(see :doc:`private_model`).

**More precisions and quantization controls.** In addition to the upstream
precisions, mbext exposes ``int2``, ``int8`` and ``int16`` (sharing the
``MatMulNBits`` pipeline) and extra options such as ``quant_weight_stats`` to
dump the distribution statistics of the quantized weights.

**End-to-end trained tests.** Slow, weight-dependent tests (gated behind
``LONGTEST=1`` in ``tests/trained``) run a real, trained tiny model through
conversion and generation to catch issues the random-weight tests cannot.

What stays the same
-------------------

* The CLI flags (``-m``, ``-i``, ``-o``, ``-p``, ``-e``, ``-c``,
  ``--extra_options``).
* The output layout (ONNX model plus ``genai_config.json`` and processing files)
  consumed by onnxruntime-genai.
* The direct-graph-construction philosophy, as opposed to exporting a traced
  PyTorch graph.
