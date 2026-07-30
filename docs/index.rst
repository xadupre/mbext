mbext documentation
====================

**mbext** (``modelbuilder``) writes large language models in the ONNX format so
they can run with `onnxruntime-genai
<https://github.com/microsoft/onnxruntime-genai>`_. The code base started as a
fork of the model builder shipped in `onnxruntime-genai
<https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models>`_
and extends it with **many more architectures** and, above all, a **fast and
short test suite**.

The source code is hosted on GitHub: `xadupre/mbext
<https://github.com/xadupre/mbext>`_.

Why mbext?
----------

The main advantage of mbext is its **short CI and fast tests**. Every supported
architecture is validated with a tiny, randomly-initialised model (typically a
single hidden layer) that:

* runs **fully offline** — no model weights are downloaded from the Hugging Face
  hub, so tests never fail because of a network hiccup or a gated repository;
* runs in **seconds**, not minutes — the models are small enough that the whole
  ``tests/fast`` suite completes quickly on a standard CI runner;
* checks **numerical discrepancies** between the PyTorch reference and the ONNX
  model, so a regression in any builder is caught immediately.

This means a contributor adding a new architecture gets fast, deterministic
feedback, and the project keeps a green CI without paying for GPU time or large
downloads. See :doc:`design` for how this is achieved.

.. toctree::
   :maxdepth: 1
   :caption: User guide

   design
   command_lines
   architectures
   private_model

.. toctree::
   :maxdepth: 1
   :caption: Comparisons

   differences_modelbuilder
   differences_mobius

.. toctree::
   :maxdepth: 1
   :caption: Examples

   auto_examples/index

Quick start
-----------

Convert ``Qwen/Qwen3-8B`` to ONNX for CPU with int4 precision:

.. code-block:: bash

    python -m modelbuilder.builder \
        -m Qwen/Qwen3-8B \
        -o qwen3-8b-cpu-int4 \
        -p int4 \
        -e cpu \
        -c cache_dir

Run the fast tests:

.. code-block:: bash

    pytest tests/fast

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
