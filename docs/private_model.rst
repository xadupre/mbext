Custom model in a separate file (``--private``)
===============================================

The ``--private`` option converts a custom model whose implementation lives
outside this package. It is meant for **private architectures** that cannot be
added to the public dispatch chain — for example a proprietary model that must
stay in your own repository.

A complete, runnable example is available in `examples/private_model
<https://github.com/xadupre/mbext/tree/main/examples/private_model>`_ (see
`A complete example`_ below).

The option value is made of up to three ``;``-separated file paths:

.. code-block:: bash

    python -m modelbuilder.builder \
        -i my-model \
        -o my-model-cpu-fp32 \
        -p fp32 \
        -e cpu \
        --private "modeling.py;convert.py;test.py"

The three paths are:

``modeling-file`` (optional, may be empty)
    Imported **before** the Hugging Face config is loaded, so a custom
    architecture can register itself with ``transformers`` (through
    ``AutoConfig.register`` / ``AutoModelForCausalLM.register``).

``convert-file``
    Defines the ONNX builder used for the conversion. The builder is the
    module-level ``MODEL_BUILDER`` attribute if present, otherwise the single
    :class:`modelbuilder.builders.Model` subclass defined in the file.

``fast-test-file`` (optional)
    A fast test file used to validate the conversion.

Running the fast tests
----------------------

If no model id is given on the command line (both ``-m/--model_name`` and
``-i/--input`` are omitted), the ``fast-test-file`` from the ``--private`` option
is executed as a script (``python fast-test-file``) instead of converting a
model:

.. code-block:: bash

    python -m modelbuilder.builder --private "modeling.py;convert.py;test.py"

The ``--private`` option must then provide a ``fast-test-file`` (the third path);
otherwise an error is raised. In this test mode ``-o``, ``-p`` and ``-e`` are not
required.

A complete example
------------------

A complete, runnable example lives in `examples/private_model
<https://github.com/xadupre/mbext/tree/main/examples/private_model>`_. It defines:

* a **modeling file** implementing a Mixture-of-Experts model from scratch
  (``PrivateConfig``, ``PrivateDecoderLayer``, ``PrivateModel``,
  ``PrivateModelForCausalLM``) plus its config and tokenizer;
* a **converter** that builds the custom MoE decoder layer;
* a **fast test** (discrepancy + genai) on a small dummy model;
* a **CI job** (``.github/workflows/private_example.yml``) that checks the
  ``--private "modeling.py;convert.py;test.py"`` command line works.

The same mechanism powers the trained example gated behind ``LONGTEST=1``
(``tests/trained/test_trained_private_moe.py``), which trains the private MoE
model, converts it with ``--private`` and validates the discrepancy and genai
generation end to end.
