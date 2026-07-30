Design
======

mbext converts a Hugging Face model into an ONNX graph that can be consumed by
`onnxruntime-genai <https://github.com/microsoft/onnxruntime-genai>`_. This page
describes the moving pieces and the conventions that make the project easy to
extend and cheap to test.

Overview
--------

The conversion is driven by a single entry point,
:func:`modelbuilder.builder.create_model`. Given a model name (or a local
folder), a target precision and an execution provider, it:

#. loads the Hugging Face ``config`` (and optionally a PEFT adapter);
#. resolves the ONNX input/output precision and the quantization dtype;
#. dispatches on ``config.architectures[0]`` to the matching **builder**;
#. builds the ONNX graph, saves it, and writes the companion files
   (``genai_config.json``, tokenizer, pre/post-processing, VS Code settings).

Builders
--------

Each architecture is implemented by a ``Model`` subclass living in
``modelbuilder/builders/``. A shared base class
(:mod:`modelbuilder.builders.base`) provides the common building blocks
(embeddings, attention, MLP, normalization, RoPE, KV-cache, Mixture-of-Experts,
quantization, ...). A concrete builder mostly overrides the few pieces that
differ for its family, for example ``make_layer``, ``make_attention`` or
``make_moe``.

The dispatch chain in :func:`~modelbuilder.builder.create_model` maps a Hugging
Face architecture string to its builder. The list is introspected at
documentation build time (see :doc:`architectures`), so it never drifts out of
sync with the code.

Precision and quantization
--------------------------

mbext supports ``fp32``, ``fp16``, ``bf16`` and integer precisions (``int2``,
``int4``, ``int8``, ``int16``). Integer precisions share the ``MatMulNBits``
pipeline; the number of bits comes from ``INT_PRECISION_BITS`` and is passed to
the quantizer through the ``int4_bits`` extra option, while the ONNX dtype stays
``INT4``/``UINT4``. A rich set of ``--extra_options`` controls block size,
accuracy level, symmetric vs. asymmetric quantization, weight sharing and more
(see :doc:`command_lines`).

Testing philosophy
------------------

The distinctive feature of mbext is how it tests conversions. Instead of
downloading real checkpoints, each fast test:

#. builds a **tiny** Hugging Face model (usually one hidden layer) with the same
   architecture but small dimensions;
#. initialises it with **random weights** (seeded for reproducibility);
#. converts it with :func:`~modelbuilder.builder.create_model`;
#. compares the ONNX outputs to the PyTorch reference within a precision-aware
   tolerance.

Because the models are tiny and offline, the suite is fast and deterministic.
Tests extend ``ExtTestCase`` from :mod:`modelbuilder.ext_test_case`, which
provides helpers such as ``run_random_weights_test``,
``run_greedy_generation_test`` and ``run_genai_generation_test``. Slow,
weight-dependent tests live in ``tests/trained`` and are gated behind
``LONGTEST=1``.

Output layout
-------------

Converting a model writes, in the output folder:

* the ONNX model (``model.onnx`` and, for large models, ``model.onnx.data``);
* ``genai_config.json`` describing how onnxruntime-genai should run the model;
* the tokenizer and pre/post-processing files copied from the source model;
* a ``.vscode/settings.json`` that excludes the large artifacts from the file
  watcher.
