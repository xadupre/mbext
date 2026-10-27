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

Fast export with onnx-light
--------------------------

mbext uses `onnx-light <https://github.com/xadupre/onnx-light>`_ as its ONNX
export backend. The shared builder constructs graphs with
``onnx_light.onnx_core.graph_builder.GraphBuilder`` and finalizes them into
onnx-light ONNX model objects. This builds the graph directly rather than
tracing a PyTorch forward pass.

Dense ``LlamaForCausalLM``, ``Qwen2ForCausalLM`` and ``Qwen3ForCausalLM``
safetensors checkpoints are bound directly to the graph builder from their
configuration and named tensors, without instantiating or traversing a
Transformers model. Floating-point exports read only safetensors headers:
initializers retain external file locations, byte ranges, shapes and original
dtypes. Casts, transposes and QKV concatenation are ONNX graph operations, not
export-time tensor operations. Source/config/output dtype conversions retain
their order, and tied input/output weights reference the same source range.
Zero-valued biases cannot be detected without reading values, so their Add
nodes are retained.

On-the-fly quantization still reads checkpoint tensors directly and uses the
existing quantization helpers; it does not instantiate a Transformers model
either. The shared safetensors reader also serves MLX loading, and standard
shard selection uses the existing checkpoint index resolver.

Normal floating-point exports copy complete, unchanged checkpoint shards into
``.weights`` to preserve the input/cache and produce an autonomous output.
This is file-level copying, not materializing tensor buffers or reserializing
weights. Separate files satisfy onnx-light's external-data single-hard-link
policy. Existing export-owned files can be replaced, but unrelated destination
collisions are rejected. If checkpoint files already live in the output's
``.weights`` directory, no checkpoint copy is performed. ``--reuse-weights``
retains its explicit ownership-transfer semantics. Shard-size limits apply to
generated constants, not to the unchanged checkpoint files.

The public CLI is unchanged. Hub checkpoints use the existing Hugging Face
snapshot cache. Other architectures, PyTorch ``.bin`` files, GGUF, prequantized
checkpoints, PEFT adapters, remote custom model classes and specialized/MLX
builders retain their existing loading paths. Private builder subclasses do
not opt into direct loading implicitly. Floating-point exports keep a named
metadata dictionary, not a weight state dictionary. Only the quantization path
materializes the checkpoint tensors.

Construction first prepares the inputs and initializers and records node
requests in a dependency-indexed queue, not in a second protobuf graph.
Finalization submits ready requests through ``GraphBuilder.make_node``,
including contrib and normalization operators. Newly inserted producers wake
only their dependent requests, including subgraph captures. Output declarations
wait until their producers exist. Unresolved values and cycles are rejected
before exporting the complete native graph; there is no Python node merge,
final topological sort or lifetime-metadata repair.

Only initializer storage preparation runs in worker threads. Registration and
node insertion remain on the calling thread; Python layer traversal is not
parallelized by using GraphBuilder.

For eligible CPU tensors, initializers borrow PyTorch storage through DLPack or
NumPy storage through ``GraphBuilder.init(copy=False)``. Avoiding unnecessary
weight copies reduces graph-construction overhead, especially for large models;
conversions that require a different dtype or layout may still copy data.

Model loading, saving and external-data sharding use ``onnx_light.onnx`` APIs.
The ``--max-shard-size`` option delegates shard creation to onnx-light, while
``--reuse-weights`` can reference existing safetensors checkpoint bytes for
floating-point exports (see :doc:`command_lines`). These mechanisms reduce
export I/O and copying; they do not change the target inference runtime:
the resulting ONNX files are consumed by onnxruntime-genai.

``examples/benchmark_zero_copy_initializers.py`` compares rewriting a checkpoint
weight (``legacy``) with referencing its existing safetensors payload
(``zero_copy``). The checkpoint is created in a separate setup process so its
allocation does not inflate the export processes' peak RSS. Its mapped tensor
is loaded before timing graph construction, file saving and checkpoint
relocation. Both modes use onnx-light's file writer;
neither builds a protobuf byte string containing the weights. The report
separates ONNX metadata size, reused payload bytes and rewritten payload bytes.
An unchanged reused weight must report zero rewritten payload bytes and no new
``.data`` file. Same-filesystem relocation preserves the checkpoint inode;
cross-filesystem relocation may still copy the original file with
``shutil.copy2``. This permitted relocation is distinct from reserializing
the weight tensors.

Direct checkpoint export measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A local comparison used pretrained ``Qwen/Qwen3-0.6B`` snapshot
``c1899de289a04d12100db370d81485cdf75e47ca`` (BF16 checkpoint), CPU FP32 output,
default QKV packing and normal external-data saving (not ``--reuse-weights``).
The baseline was updated ``main`` at ``a1a23dc``. Both checkouts used the same
cached snapshot, Python 3.12, PyTorch ``2.16.0.dev20261006+cu132``,
Transformers 5.15.0, and an i7-13800H under Linux/WSL with
``OMP_NUM_THREADS=2`` and ``MKL_NUM_THREADS=2``.

Each export ran ``create_model`` in a fresh subprocess with offline Hub access.
After one warm-up per checkout, three sequential samples per checkout alternated
order, retaining warm filesystem caches. The following are medians, with ranges
for process wall time and process-wide peak RSS:

========================= ===================== =====================
Measurement               Main                  External checkpoint
========================= ===================== =====================
Complete process (s)      22.11 (20.52--23.58)    25.98 (24.87--27.13)
Imports (s)               9.676                 8.062
Checkpoint loading (s)    1.224                 0.003
Graph preparation (s)     1.522                 0.193
Finalization/saving (s)   7.499                 15.280
Peak RSS (MiB)            8333 (8289--8334)      1257 (1257--1258)
========================= ===================== =====================

Phase timers wrapped ``Model.load_weights``, ``Model.make_model`` (subtracting
loading), and ``Model.save_model``. Imports were timed before importing the
public builder; the parent measured full subprocess lifetime, including startup
and teardown. Remaining time includes configuration, tokenizer and pipeline
work. RSS came from Linux ``resource.getrusage(RUSAGE_SELF).ru_maxrss``.
Pretrained ONNX Runtime logits were bit-identical; copied checkpoint bytes were
identical. Normal exports retained 1,503,300,328 checkpoint bytes and wrote
20,971,520 bytes of generated external constants, rather than rewriting the
baseline's 2,405,142,528 external-data bytes.

These are small, noisy samples on a shared machine, not a general speedup
guarantee. The external path substantially reduced export RSS but did not
improve total time in this comparison. Import differences are not attributable
to removing model instantiation.

A separate instrumented export measured ``shutil.copy2`` at 0.788 s for the
1.5 GB checkpoint, native finalization at 12.475 s and complete
finalization/saving at 13.367 s. The saving timer must not be mistaken for
checkpoint copying. Python 3.12 used its Linux ``sendfile`` fast path.

To measure a valid no-copy export, the same checkpoint was placed in the
output's ``.weights`` directory before timing. One warm-up and three fresh
processes, with guards rejecting checkpoint materialization and copying, gave
a median complete process time of 21.96 s (21.74--23.49), imports of 7.044 s,
graph preparation of 0.170 s, native finalization of 12.210 s and
finalization/saving of 12.353 s. Source placement was setup, not an export-time
copy. Thus the remaining cost is not checkpoint copying.

Graph preparation only queues Python node requests: actual native insertion
and export happen during finalization. A profile attributed about 9 s to
native node insertion and 3.3 s to native model export. The unchanged baseline
finalized in 0.916 s in a separate profile. These transformed external-weight
graphs still expose a native finalization bottleneck; the small preparation
time is not the time for constructing the complete graph.

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
