Supported architectures
=======================

The table below lists every Hugging Face architecture that
:func:`modelbuilder.builder.create_model` can convert, together with the builder
class that handles it. It is generated automatically at documentation build time
by parsing the dispatch chain of ``create_model`` (see
:func:`modelbuilder.architectures.list_supported_architectures`), so it is always
in sync with the code.

The example model IDs are a static, non-exhaustive selection favoring the model
families that introduced each architecture in Transformers. An empty cell means
no example is recorded, not that the architecture is unsupported. These examples
do not guarantee support for every checkpoint variant, modality, or export
option; the export restrictions documented below still apply.

To obtain the IDs without downloading models or importing Transformers:

.. code-block:: python

    from modelbuilder.architectures import (
        list_supported_architectures,
        list_supported_model_ids,
    )

    model_ids = list_supported_model_ids()
    for architecture in list_supported_architectures():
        print(architecture.name, architecture.model_ids)

The architecture name is the value of ``config.architectures[0]`` in the Hugging
Face ``config.json``. To convert a model of a given family, point ``-m`` or
``-i`` at a checkpoint whose config declares that architecture.

.. supported-architectures::

DeepSeek-V4.1-Flash
------------------

``deepseek-ai/DeepSeek-V4.1-Flash`` declares ``DeepseekV41ForCausalLM``.
It is not a DeepSeek-V3 architecture alias. The export includes both the
language decoder and the vision encoder/projector by default, with a
``deepseek_v41_config.json`` manifest describing their tensor interfaces.
The released checkpoint's FP8 matrices and packed FP4 experts are decoded
before the requested export quantization is applied.

V4.1 carries sliding-window caches, compressed shared KV/indexer caches,
incomplete compression groups, and integer Engram history. Use its explicit
state driver rather than a conventional per-layer K/V decoder wrapper:

.. code-block:: python

    import numpy as np
    from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session

    session = DeepseekV41Session(output_dir, backend="onnxruntime")
    tokens = session.generate(
        np.array([[4, 5, 6]], dtype=np.int64), max_new_tokens=10
    )

The following matrix describes offline tiny-model validation, not inferred
support from successful export alone. Each numerical/generation entry covers
both text and image prompts, prefill and cached decoding.

.. list-table:: DeepSeek-V4.1 precision/backend coverage
   :header-rows: 1
   :widths: 10 24 24 24 18

   * - Precision
     - ORT CPU
     - ORT CUDA
     - onnx-light-cpu
     - Native onnx-light
   * - FP32
     - Numerical + generation
     - Not validated
     - Numerical + generation
     - Empty-cache ``Pow`` blocked
   * - FP16
     - Numerical; text token parity
     - Not validated
     - Not validated
     - Empty-cache ``Pow`` blocked
   * - INT4
     - Numerical + generation
     - Not validated
     - Numerical + generation
     - ``MatMulNBits`` unavailable
   * - BF16
     - Arithmetic unsupported
     - Numerical + generation
     - ``Where`` unsupported
     - ``Where`` unsupported

INT4 tests verify packed 4-bit ``MatMulNBits`` weights in both exported
components. Greedy tokens must match between ORT CPU and onnx-light-cpu running
the same quantized graph; independent FP32 PyTorch logits bound quantization
error. Quantization is not assumed to preserve unquantized FP32 argmax.
Floating FP4 cache values can cross bin boundaries between kernels, so integer
history is checked exactly while logits use precision-appropriate tolerances.

BF16 CUDA tests use real BF16 inputs, weights and recurrent caches. Text greedy
tokens are compared exactly with the BF16 PyTorch reference. Image generation
can diverge at BF16/FP4 rounding boundaries: every generated prefix is replayed
against independent PyTorch logits, checking numerical discrepancies and that
the selected token is compatible with the reference error bound.
Without CUDA the full BF16 generation test is explicitly skipped; export and
the BF16 runtime binding/lifetime regression still run on CPU.

For BF16 CUDA execution, select the provider explicitly:

.. code-block:: python

    session = DeepseekV41Session(
        output_dir, providers=["CUDAExecutionProvider"]
    )

The driver uses ORT I/O binding for BF16 because its ordinary NumPy bridge does
not accept ``ml_dtypes.bfloat16``. Returned state remains CPU NumPy arrays;
this is not a GPU-resident-cache implementation or throughput claim.
ORT CPU BF16 arithmetic is tracked in `microsoft/onnxruntime#28768
<https://github.com/microsoft/onnxruntime/issues/28768>`_.
Both onnx-light backends currently reject BF16 ``Where``
(`xadupre/onnx-light#5998 <https://github.com/xadupre/onnx-light/issues/5998>`_).
Native empty-cache ``Pow`` broadcasting is tracked in `xadupre/onnx-light#5993
<https://github.com/xadupre/onnx-light/issues/5993>`_; it also blocks testing
portable INT4 QDQ graphs without the CPU contrib kernels.

For image prompts, ``session.encode_image(image)`` accepts a local Pillow
image or RGB NumPy array and returns its token types and embeddings.
Expand each image placeholder to that many copies of ``image_token_id``;
place the returned types and embeddings into the corresponding prompt
positions. Use ``-1`` token types for text, and pass the combined
``token_types [batch, sequence]`` and
``inputs_embeds [batch, sequence, hidden_size]`` to ``generate``.
Image start, end and row-delimiter embeddings are included.

Positions must start at zero and remain contiguous and synchronized across
the batch; padding and speculative/MTP heads are not supported.
Prefill attention materializes dense matrices within each submitted chunk.
For long prompts, call ``run`` on consecutive short chunks while passing
the returned state, rather than prefilling the whole prompt through ``generate``.
Reduced precision and low-bit cache rounding can change greedy token choices;
the precision-specific numerical and generation guarantees are listed above.

Tests with ONNX Runtime GenAI 0.19.0 and 0.20.0 fail when updating these
heterogeneous state inputs across generation steps; native integration is tracked in
`issue #604 <https://github.com/xadupre/mbext/issues/604>`_.
No misleading GenAI configuration is emitted. Adapter export,
``--reuse-weights`` and layer-count overrides are not supported.
The original model is very large: random-weight offline tests validate
numerics and cached generation but do not constitute a full-checkpoint
memory or throughput benchmark.

Kimi K3.1 Pro
-------------

``moonshotai/Kimi-K3.1-Pro`` declares
``KimiK31ForConditionalGeneration``. Its KDA recurrence, positional-encoding-free
gated DSA, block attention residuals, latent SiTU MoE, and MoonViT/PatchMergerV2
are exported directly; this is not a Kimi K2 or DeepSeek architecture alias.
Native BF16 and compressed-tensors MXFP4 checkpoint weights are supported.
Source MXFP4 weights are decoded before optional ONNX weight quantization.

.. code-block:: bash

    python -m modelbuilder.builder \
        -m moonshotai/Kimi-K3.1-Pro -o kimi-onnx -p int4 -e cpu -c cache

Use the explicit-state driver, not a conventional per-layer K/V cache wrapper:

.. code-block:: python

    import numpy as np
    from modelbuilder.helpers.kimi_k31_runtime import KimiK31Session

    session = KimiK31Session("kimi-onnx", backend="onnxruntime")
    tokens = session.generate(
        np.array([[4, 5, 6]], dtype=np.int64), max_new_tokens=10
    )

The decoder graph advances one token at a time; ``run`` and ``generate``
prefill arbitrary prompt lengths sequentially. This is a correctness-first
implementation, not a parallel-prefill or throughput-optimized exporter.
Positions must be contiguous, synchronized across batch rows, and start at
zero. Padding, packed documents, adapters, tied embeddings, weight reuse,
extra hidden-state outputs, speculative/MTP heads and layer-count overrides
are not supported. Change a tiny test config and its one-based attention
schedule together instead.

For images and video, use the checkpoint's processor for prompt construction,
resizing, normalization, frame sampling, timestamp tokens and temporal
grouping. ``session.encode_media(patches, grid_shape)`` accepts one processed
image or frame group: normalized RGB patches of shape
``[time * height * width, 3, patch_size, patch_size]`` in time/row/column
order and ``grid_shape=[time, height, width]``. Spatial grid dimensions must be
divisible by the merge kernel. Time cannot exceed ``init_pos_emb_time``;
encode multiple images/groups separately. The encoder pools time and returns
``[media_tokens, hidden_size]``. Insert these features at expanded media
placeholder positions, pass ``token_types=1`` there (``-1`` for text), and
provide aligned ``inputs_embeds [batch, sequence, hidden_size]`` to generation.
Image/video delimiters and timestamp tokens remain ordinary text embeddings.
This does not provide raw-file image/video preprocessing in GenAI.

Offline seeded tests cover text, images and temporal frame groups, dynamic
batches/grids, prefill, growing state, cached decode and greedy generation:

* FP32: numerical and exact token parity with PyTorch on ORT CPU, native
  onnx-light and onnx-light-cpu.
* FP16: numerical comparisons and generated-prefix/token-error checks on
  all three CPU backends.
* INT4: real packed weights in decoder and vision graphs, independent FP32
  numerical error bounds, and exact generated-token comparisons between
  ORT and onnx-light-cpu. Native onnx-light uses the standard INT4 QDQ
  alternative (``--extra_options use_qdq=true``), not ``MatMulNBits``.
  The released model's widths are block aligned; native QDQ with partial
  final blocks is affected by `xadupre/onnx-light#6009
  <https://github.com/xadupre/onnx-light/issues/6009>`_.
* BF16: real ORT CUDA execution and generated-prefix/token-error checks.
  State uses CPU NumPy buffers with bit-preserving BF16 bindings, not
  GPU-resident caches. ORT CPU arithmetic remains limited by
  `microsoft/onnxruntime#28768
  <https://github.com/microsoft/onnxruntime/issues/28768>`_; BF16 ``Where``
  on both light backends is tracked in `xadupre/onnx-light#5998
  <https://github.com/xadupre/onnx-light/issues/5998>`_.

Exact DSA-score ties use ONNX's lower-index-first rule, which is not guaranteed
by PyTorch's ordinary ``topk``. Quantization and reduced precision can change
near-tie selection and greedy tokens; full-precision argmax is not promised
for those exports.

``kimi_k31_config.json`` describes the explicit state and both components.
No GenAI configuration is emitted: reliable heterogeneous recurrent-state
rebinding is blocked by `issue #604
<https://github.com/xadupre/mbext/issues/604>`_.
The 2.8T-parameter checkpoint has not been exported end to end here.
All decoded weights must currently fit RAM during conversion; external-data
sharding does not eliminate that memory requirement. The offline tests make
no full-checkpoint memory, quality or performance claim.
The tracing path uses Torch's private TorchScript exporter and supports
ONNX opsets 20--23 (default 23); INT4 QDQ needs at least opset 21.
All ONNX loading, saving and sharding uses onnx-light.

SmolVLM image inputs
-------------------

``HuggingFaceTB/SmolVLM-256M-Instruct`` exports its text decoder by default.
Set ``multimodal=True`` (CLI: ``--extra_options multimodal=true``) to also
export ``vision_encoder.onnx`` and ``embedding.onnx``.
Adapter export and ``--reuse-weights`` are not supported for SmolVLM.

The vision encoder takes one normalized, square crop at the checkpoint's
configured image size as ``pixel_values [1, channels, height, width]``.
Use the Hugging Face processor for image splitting, resizing, normalization
and prompt construction. Masked padding and variable-size crops are not
supported by this fixed-size vision export. Encode each crop separately
and concatenate ``image_features`` in prompt order.

Pass ``input_ids [1, sequence]`` and ``image_features [image_tokens, hidden]``
to the embedding model; it replaces the image placeholders in place.
Feed its ``inputs_embeds`` output to the decoder. During cached decoding,
pass an empty ``[0, hidden]`` image-feature array to the embedding model.

ONNX Runtime GenAI can run the three components using the exported
``phi3v`` configuration and explicit preprocessed inputs: supply
``pixel_values`` and ``num_image_tokens`` through ``NamedTensors``, then
append the expanded prompt tokens. This does not provide a SmolVLM-specific
raw-image processor in GenAI.

Qwen2.5-Omni video inputs
------------------------

Export with ``multimodal=True`` to produce the shared image/video vision
encoder, audio encoder, embedding model, and thinker decoder. For direct
ONNX Runtime inference with multimodal temporal/spatial positions, also set
``use_3d_position_ids=True`` (CLI:
``--extra_options multimodal=true use_3d_position_ids=true``).
The default decoder still accepts 2-D position IDs for existing callers.
With the 3-D option, vision ``frame_ids`` and ``window_ids`` are required
inputs; without it, they retain their single-frame/window defaults.

The Hugging Face processor supplies ``pixel_values_videos`` and
``video_grid_thw``. Prepare the vision feeds as follows:

.. code-block:: python

    from modelbuilder.helpers.vision_helper import prepare_qwen25_omni_vision_inputs

    feeds = prepare_qwen25_omni_vision_inputs(
        pixel_values_videos, video_grid_thw, thinker_config.vision_config
    )
    video_features = vision_session.run(["image_features"], feeds)[0]

The helper accepts NumPy arrays in processor patch order. It computes vision
RoPE frequencies and frame/window IDs, preventing attention across separate
frames or spatial windows. A temporal grid entry counts groups of
``temporal_patch_size`` raw frames. The same helper supports images and
multiple videos with different spatial dimensions.

Pass the resulting features to ``embedding.onnx`` through ``image_features``;
this existing input is shared by images and videos. For a mixed prompt, arrange
the feature rows in the order that image/video placeholders occur in
``input_ids``. ``audio_features`` is unchanged; use an empty
``[0, hidden_size]`` array when audio is absent. Modality token IDs come from
the checkpoint config, including ``video_token_id`` / ``video_token_index``.

Feed the embedding output to ``model.onnx``. With ``use_3d_position_ids=True``,
provide ``position_ids`` of shape ``[3, batch, sequence]``, computed using
the thinker's ``get_rope_index`` with the original grids, video timing
(``second_per_grids``), and any audio metadata. Subsequent decoding positions
must account for the returned RoPE deltas.

ONNX Runtime GenAI video generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video-only prompts can also run through ONNX Runtime GenAI 0.15.2 or later
with explicit preprocessing. Export with ``use_3d_position_ids=True`` and
overlay the default ``phi3v`` configuration with Qwen-VL position handling:

.. code-block:: python

    import json
    import numpy as np
    import onnxruntime_genai as og

    config = og.Config(output_dir)
    config.overlay(json.dumps({
        "model": {
            "type": "qwen2_5_vl",
            "image_token_id": thinker_config.image_token_id,
            "video_token_id": thinker_config.video_token_id,
            "vision_start_token_id": thinker_config.vision_start_token_id,
            "vision": {"tokens_per_second": thinker_config.position_id_per_seconds},
        }
    }))
    model = og.Model(config)
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=input_ids.size + 3)
    generator = og.Generator(model, params)
    tensors = og.NamedTensors()
    for name, value in feeds.items():
        tensors[name] = value
    merge = thinker_config.vision_config.spatial_merge_size
    tensors["num_image_tokens"] = np.array(
        [np.prod(video_grid_thw, axis=1).sum() // merge**2], dtype=np.int64
    )
    tensors["video_grid_thw"] = np.asarray(video_grid_thw, dtype=np.int64)
    tensors["second_per_grid_ts"] = np.asarray(second_per_grids, dtype=np.float32)
    generator.set_inputs(tensors)
    generator.append_tokens(input_ids)
    while not generator.is_done():
        generator.generate_next_token()

GenAI runs the vision encoder, embedding model, and cached decoder; it does
not merely consume externally computed logits. The native 3-D export makes
frame/window inputs required because GenAI does not forward overridable
initializer inputs. The fp32/fp16 ``test_video_*_cpu_genai`` tests compare
positions, embeddings, logits, and generated tokens against PyTorch.

This explicit-input setup does not provide automatic raw-video processing
or Omni audio/video interleaving. GenAI's Qwen-VL positions round temporal
coordinates whereas Omni truncates them; use timings for which
``second_per_grids * position_id_per_seconds`` is integral for parity.
Use direct ONNX Runtime and Omni's ``get_rope_index`` for other timings or
audio/video interleaving. The default ``phi3v`` configuration is unchanged.

Adding a new architecture
-------------------------

Support for a new family is added by:

#. implementing a ``Model`` subclass in ``modelbuilder/builders/`` (usually
   subclassing an existing builder and overriding the few pieces that differ);
#. adding an ``elif config.architectures[0] == "<Name>ForCausalLM":`` branch to
   :func:`~modelbuilder.builder.create_model` that instantiates the builder;
#. adding a fast test under ``tests/fast`` that converts a tiny random-weight
   model and checks the discrepancies.

Once the branch is added, this page picks up the new architecture
automatically.
