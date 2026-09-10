Command lines
=============

mbext is used through its command line, ``python -m modelbuilder.builder``. This
page documents the available options.

Converting a model
-------------------

.. code-block:: bash

    python -m modelbuilder.builder \
        -m Qwen/Qwen3-8B \
        -o qwen3-8b-cpu-int4 \
        -p int4 \
        -e cpu \
        -c cache_dir

Main arguments
--------------

``-m``, ``--model_name``
    Model name on the Hugging Face hub. Do not use together with ``-i/--input``.

``-i``, ``--input``
    Path to a local folder containing the Hugging Face ``config``, model and
    tokenizer, or the path to a float16/float32 GGUF file.

``-o``, ``--output``
    Folder where the ONNX model and the additional files are written.

``-p``, ``--precision``
    Precision of the model. One of ``int2``, ``int4``, ``int8``, ``int16``,
    ``bf16``, ``fp16`` or ``fp32``.

``-e``, ``--execution_provider``
    Execution provider to target: ``cpu``, ``cuda``, ``dml``, ``webgpu`` or
    ``NvTensorRtRtx``.

``-c``, ``--cache_dir``
    Cache directory for Hugging Face files and temporary ONNX external data
    files. Defaults to ``./cache_dir``.

``-r``, ``--reuse-weights``
    For floating-point safetensors models, reference the checkpoint bytes
    directly as ONNX external data instead of copying the weights. See
    :ref:`reuse-weights`.

``--private``
    Convert a custom model implemented in separate files. See
    :doc:`private_model`.

``--extra_options``
    Space-separated ``KEY=VALUE`` pairs controlling advanced behaviour
    (quantization block size, accuracy level, weight sharing, LoRA adapter,
    number of layers, and many more).

Valid precision / execution provider combinations are: FP32 CPU, FP32 CUDA,
FP16 CUDA, FP16 DML, BF16 CUDA, FP16 TRT-RTX, BF16 TRT-RTX, INT4 CPU, INT4 CUDA,
INT4 DML and INT4 WebGPU.

.. _reuse-weights:

Reusing downloaded weights
--------------------------

``--reuse-weights`` saves disk space by referencing byte ranges in the original
``.safetensors`` files as ONNX external data. With ``--model_name``, the
checkpoint is downloaded to ``<output>/.weights``; keep that directory with the
exported model. Any initializers derived by the builder are still written to
``model.onnx.data``.

The option is only supported for floating-point exports, because quantization
changes the weight representation. If using ``--input``, the checkpoint must be
inside the output directory and its ``.safetensors`` files must not be symbolic
links. When the checkpoint dtype differs from the requested ONNX dtype, the
export contains ONNX ``Cast`` nodes rather than a copied, converted checkpoint.

.. code-block:: bash

    python -m modelbuilder.builder \
        -m Qwen/Qwen3-8B \
        -o qwen3-8b-cuda-fp16 \
        -p fp16 \
        -e cuda \
        --reuse-weights

Full help
---------

The authoritative and always up-to-date reference is the command's own help,
which lists every ``--extra_options`` key with its description:

.. code-block:: bash

    python -m modelbuilder.builder --help

Running the private fast tests
------------------------------

When neither ``-m/--model_name`` nor ``-i/--input`` is provided, the
``fast-test-file`` from the ``--private`` option is executed as a script instead
of converting a model:

.. code-block:: bash

    python -m modelbuilder.builder --private "modeling.py;convert.py;test.py"

In this mode ``-o``, ``-p`` and ``-e`` are not required. See :doc:`private_model`.
