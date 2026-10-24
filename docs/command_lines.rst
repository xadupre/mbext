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

Multi-model artifacts
---------------------

A multi-component export automatically adds **pipeline.onnx**, an onnx-light
``PipelineProto``, beside the standalone models and ``genai_config.json``.
It is an additional artifact, not a merged graph or a replacement GenAI config.
Single-model and config-only exports are unchanged.

The pipeline embeds the saved embedding, prefill and (when separate) decode
ModelProtos. A decoder used for both phases occupies only ``prefill``.
Transferred tensors keep their declared names; cache input/output pairs use
``GraphProto.persistent_bindings``. Available sampling settings and token IDs
come from the GenAI configuration; greedy decoding uses temperature zero.
The CLI reports the pipeline filename when it is saved.

No weights are loaded or externalized again. For example::

    export/
      pipeline.onnx
      embedding.onnx
      model.onnx
      embedding.onnx.data
      model.onnx.data.0
      model.onnx.data.1
      .weights/model.safetensors
      genai_config.json

Both artifacts reference the same external-data locations, offsets and lengths,
including quantized storage, ``--max-shard-size`` shards and reused source files
under ``.weights/``. Keep component protos beside ``pipeline.onnx`` and relocate
the whole output folder, including relative weight subdirectories, as a unit.
Re-exporting overwrites the pipeline without creating pipeline-specific shards.

Load the bundle explicitly as a pipeline, not with ``load_model``::

    from onnx_light.onnx import PipelineProto
    from onnx_light.onnx_proto.verify import verify_pipeline

    pipeline = PipelineProto()
    pipeline.ParseFromFile("export/pipeline.onnx")
    verify_pipeline(pipeline)

Runtime support is distinct from serialization: onnx-light currently rejects
execution of a separate ``decode`` model. Such bundles can be loaded and
validated, and their recovered components executed independently; integrated
generation must wait for runtime support. The offline tests compare generation
and state updates from recovered and standalone components using available kernels.

The current onnx-light schema supports only ``embedding``, ``prefill`` and
``decode``. Full Mistral/Phi/Qwen vision/audio exports and Whisper's audio
encoder therefore need additional component ModelProto fields (vision, speech
or encoder) and tensor/state interface support in onnx-light. These exports
retain every standalone model and GenAI configuration but explicitly report
that no complete PipelineProto can be written; they never emit a bundle that
silently drops a component. Multiple EOS IDs also require a schema extension;
the pipeline currently uses the first ID and reports this limitation.

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
``.safetensors`` files as ONNX external data. With ``--input``, checkpoint weight
shards outside the export are **moved** into ``<output>/.weights``: ownership is
transferred and the source weight files are removed. Unrelated checkpoint files
are not moved. The original directory is no longer a complete checkpoint.
With ``--model_name``, the checkpoint is downloaded to ``<output>/.weights``.
Files already inside the export are not moved again. Safetensors must be regular
files, not symbolic links; destination collisions are rejected, not overwritten.
Relocation publishes a complete destination file before removing its source,
including across filesystems, and preserves source permissions and timestamps.

Both floating-point and integer-precision exports can reuse compatible original
tensor storage. Original shards are never rewritten or resharded, and offsets
include the safetensors header. Quantization, repacking, and other transformations
that change the representation produce new payloads in a separate generated
external-data file (``model.onnx.data`` for a single model). Constants smaller
than 1024 bytes stay inline. No empty generated-weight file is created.
``--max-shard-size`` applies only to generated weights. Multi-model exports and
their additional ``pipeline.onnx`` share the original shards and deduplicated
generated storage. Each component writes only its new generated payloads through
onnx-light external storage; component protos never stage those bytes inline.
Finalization also reuses this generated-storage index for additional components
discovered through ``genai_config.json``. New payloads discovered at that stage
use fresh onnx-light-assigned external-data filenames, preserving the contents
and offsets of all previously referenced shards.

Use a clean output directory for each independent export. Shared-weight
finalization does not prune existing generated data files, because other
components may still reference them.

Configuration-only exports do not inspect or relocate checkpoint weight files,
even when ``--reuse-weights`` is enabled.
Otherwise, a missing safetensors checkpoint is reported before relocation.

Keep the entire export directory together when relocating it: relative external
locations do not depend on the old checkpoint path or the working directory.
When a reusable checkpoint dtype differs from the requested ONNX dtype, the
export contains ONNX ``Cast`` nodes rather than a copied, converted checkpoint;
transpose views can similarly reuse the original bytes through ``Transpose``.

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
