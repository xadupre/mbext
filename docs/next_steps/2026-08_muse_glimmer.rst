.. _l-next-steps-muse-glimmer:

Support for Muse Glimmer 30B
============================

:Date: 2026-08

**discussion**

Objective
+++++++++

Add complete support for
`meta-models/Muse-Glimmer-30B
<https://huggingface.co/meta-models/Muse-Glimmer-30B>`_:

* text-only and vision-language inference;
* prefill, incremental decoding, and generation with a KV cache;
* CPU, CUDA, DML, WebGPU, and TensorRT RTX execution providers whenever
  supported by ONNX Runtime;
* FP32, FP16, BF16, and weight-only integer quantization modes exposed by
  ``mbext``;
* fast offline tests with random weights and trained tests using the released
  checkpoint.

The implementation should produce the three-model layout expected by ONNX
Runtime GenAI:

.. code-block:: text

    vision_encoder.onnx
    embedding.onnx
    model.onnx
    genai_config.json

The text-only path should remain usable independently of the vision encoder
and embedding mixer.

Architecture
++++++++++++

Text decoder
^^^^^^^^^^^^

The dense text decoder has 52 layers, hidden size 6656, intermediate size
19968, 32 query heads, 2 key/value heads, and head size 128. It requires the
following model-specific behavior:

* a repeating ``[sliding, sliding, sliding, full]`` attention pattern;
* a local attention window of 2048 tokens;
* RoPE with theta 500000 on sliding-attention layers and no positional
  embedding on full-attention layers;
* scale-free RMS normalization of Q and K, followed by a query multiplier of
  3.87;
* a sigmoid attention gate applied before the output projection;
* four centered RMSNorm operations per decoder layer, with the checkpoint
  weight interpreted as ``1 + weight``;
* epsilon 1e-5 for input and pre-feed-forward norms and 1e-8 for
  post-attention and post-feed-forward norms;
* a scale-free RMSNorm after token embedding;
* an output multiplier of 0.19611613513818404 before the final logit softcap
  of 20.

The model is dense and does not need the ``MoE`` or ``QMoE`` operators.

Vision pipeline
^^^^^^^^^^^^^^^

The perception encoder is a 50-layer ViT-G/14 with hidden size 1536,
intermediate size 8960, 16 heads, and alternating window/full attention. The
export must include:

* spatial patch size 14 and temporal patch size 2;
* dynamic image grids and two-dimensional positional embeddings;
* window and full vision attention;
* 2x2 patch merging;
* the ``6144 -> 4096 -> 4096 -> 6656`` multimodal projection;
* replacement of ``<|patch|>`` token embeddings with image features;
* rectangular images, multiple images, and dynamic visual token counts.

Transformers dependency
+++++++++++++++++++++++

The released configuration names ``MuseGlimmerForConditionalGeneration`` and
was produced with ``transformers==5.15.0.dev0``. The dependency and fast-test
matrix now use Transformers 5.15.0, which contains
``transformers.models.muse_glimmer``.

Builder implementation
++++++++++++++++++++++

Add ``modelbuilder/builders/muse_glimmer.py`` with separate builders for the
text decoder, vision encoder, embedding mixer, and conditional-generation
wrapper.

The text builder can reuse the generic GQA, KV-cache, MLP, quantization, and
logit-softcap helpers. It needs model-specific overrides for:

* layer-dependent sliding attention and RoPE/NoPE selection;
* scale-free per-head Q/K normalization;
* the query scale and sigmoid attention gate;
* the four-norm residual block;
* normalized token embeddings and output multiplier;
* loading ``MuseGlimmerForConditionalGeneration`` weights.

The conditional wrapper should save all three ONNX graphs, processing files,
and a ``muse_glimmer`` ONNX Runtime GenAI configuration. A text-only option
should export the decoder without requiring the vision graph.

Fast tests
++++++++++

Add ``tests/fast/test_random_muse_glimmer.py``. Tests must construct tiny
random-weight models locally and must not download the 30B checkpoint.

Text tests
^^^^^^^^^^

Use at least four decoder layers so that one model covers sliding attention,
full attention, RoPE, and NoPE. Compare PyTorch and ONNX Runtime for:

1. normalized token embeddings;
2. Q/K normalization, query scaling, and sigmoid gating;
3. both norm epsilon values and all four norm positions;
4. prefill logits and returned KV caches;
5. one-token decoding using the prefill caches;
6. output multiplication and logit softcapping;
7. deterministic greedy generation.

The graph checks should confirm the absence of rotary nodes on NoPE layers and
the local window on sliding layers.

Multimodal tests
^^^^^^^^^^^^^^^^

Use a reduced vision configuration that still contains window and full
attention. Validate:

1. numerical parity of the vision encoder;
2. dynamic square and rectangular patch grids;
3. patch merging and projector output;
4. single-image and multi-image embedding replacement;
5. the complete vision-to-embedding-to-decoder pipeline;
6. the three exported graph signatures and ``genai_config.json``;
7. text-only use of the conditional-generation checkpoint.

Mode matrix
^^^^^^^^^^^

The tests should cover every precision/provider combination advertised as
supported by the command line, rather than assuming that every Cartesian
combination is valid.

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Provider
     - Required modes
     - Validation
   * - CPU
     - FP32, INT4
     - Prefill, decode, generation, and multimodal pipeline
   * - CUDA
     - FP32, FP16, BF16, INT4
     - Prefill, decode, generation, multimodal, and CUDA Graph
   * - DML
     - FP16, INT4
     - Build, session creation, prefill, and decode
   * - WebGPU
     - INT4
     - Build and WebGPU graph-compatible model structure
   * - TensorRT RTX
     - FP16, BF16
     - Build, session creation, prefill, and decode

INT2, INT8, and INT16 use the common ``MatMulNBits`` pipeline in ``mbext``.
They should receive focused build and numerical tests before being documented
as supported Muse Glimmer modes. FP16 on CPU may remain an internal regression
test but should not be advertised unless the global CLI support contract is
updated.

Trained tests
+++++++++++++

Add ``tests/trained/test_trained_muse_glimmer_30b.py`` guarded by
``@long_test()``. The checkpoint is approximately 60 GB, so these tests need a
dedicated large-memory runner and must reuse cached downloads and exports.

The trained suite should contain:

* BF16 CUDA prefill and one-token decode parity;
* BF16 CUDA deterministic greedy generation;
* INT4 CUDA text generation and image-conditioned generation;
* INT4 CPU generation when sufficient RAM is available;
* golden logits for prefill and decode;
* exact first-token agreement and a stable generated-token prefix;
* vision-feature parity for at least one fixed image;
* CUDA Graph generation with a shared KV-cache buffer.

Provider-specific trained tests should be enabled only after the corresponding
fast mode is green. Expensive exports should be created once per
precision/provider pair and shared by discrepancy and generation tests.

ONNX Runtime dependencies and gaps
++++++++++++++++++++++++++++++++++

ONNX Runtime core
^^^^^^^^^^^^^^^^^

No new ONNX Runtime operator is known to be required for the CUDA
implementation. A complete INT4 CUDA package has already been exported and
validated with ONNX Runtime 1.28. Q/K normalization, sigmoid gating, NoPE, and
the output softcap can be represented around existing attention operators.

The main risk is provider coverage. DML, WebGPU, and TensorRT RTX have not been
validated publicly for the complete model. Their attention,
``RMSNormalization``, scatter, and dynamic vision paths need explicit session
tests. Unsupported fused paths should use existing primitive ONNX
decompositions where practical rather than silently changing model behavior.

ONNX Runtime GenAI
^^^^^^^^^^^^^^^^^^

Native multimodal loading currently depends on
`onnxruntime-genai pull request 2397
<https://github.com/microsoft/onnxruntime-genai/pull/2397>`_. As of 2026-08-14,
that pull request is open. It adds:

* ``muse_glimmer`` to the vision-language model registry;
* use of the packed Qwen image preprocessing path while retaining ordinary
  one-dimensional decoder positions;
* expansion of each ``<|patch|>`` placeholder to the visual token count.

The validated package requires ONNX Runtime 1.28 and ONNX Runtime GenAI
0.16.0-dev built with that change. The latest stable
``onnxruntime-genai==0.15.2`` pinned by ``mbext`` remains insufficient for
native multimodal execution.

Video processing is not covered by that pull request. It should not be
declared supported by the ORT GenAI pipeline until token expansion,
preprocessing, and end-to-end generation are implemented and tested.

Speculative decoding
^^^^^^^^^^^^^^^^^^^^

The optional DFlash drafter predicts blocks of 16 tokens and is not a standard
autoregressive draft model. ONNX Runtime GenAI does not currently expose the
required block-diffusion speculative orchestration. Base-model generation
must be completed independently; DFlash support is a separate follow-up.

Implementation order
++++++++++++++++++++

1. Update the Transformers test dependency and add architecture dispatch.
2. Implement and test the tiny text decoder in FP32 on CPU.
3. Add prefill, decode, cache, softcap, and greedy-generation parity.
4. Add integer quantization and the CUDA precision modes.
5. Implement the vision encoder, projector, and embedding mixer.
6. Add the complete tiny multimodal pipeline and GenAI configuration.
7. Validate DML, WebGPU, and TensorRT RTX and document only green modes.
8. Add cached BF16 and INT4 trained tests on large-memory runners.
9. Enable native ORT GenAI multimodal tests once pull request 2397 is
   available in the selected dependency.
10. Track video and DFlash as separate runtime features.

References
++++++++++

* `Muse Glimmer model card
  <https://huggingface.co/meta-models/Muse-Glimmer-30B>`_
* `Muse Glimmer configuration
  <https://huggingface.co/meta-models/Muse-Glimmer-30B/raw/main/config.json>`_
* `Transformers implementation
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/muse_glimmer/modeling_muse_glimmer.py>`_
* `Mobius export support
  <https://github.com/onnxruntime/mobius/pull/475>`_
* `ONNX Runtime GenAI integration
  <https://github.com/microsoft/onnxruntime-genai/pull/2397>`_
* `Validated ONNX INT4 CUDA package
  <https://huggingface.co/justinchuby/Muse-Glimmer-30B-ONNX-INT4-CUDA>`_
