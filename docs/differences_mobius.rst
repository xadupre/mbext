Differences with Mobius (onnxruntime/mobius)
============================================

`Mobius <https://github.com/onnxruntime/mobius>`_ is a newer model builder for
onnxruntime-genai. Like mbext, it constructs ONNX graphs declaratively and
applies Hugging Face or custom weights, rather than tracing an existing PyTorch
graph. Both projects target the same runtime (onnxruntime-genai) and overlap on
the same kinds of models (LLMs, MoE, multimodal, ...).

How Mobius is organised
-----------------------

Mobius builds the ONNX graph with the ``onnxscript.nn`` API
(`onnxscript <https://github.com/microsoft/onnxscript>`_), composing
``onnxscript.nn.Module`` objects the way one would compose ``torch.nn.Module``
objects. Its pipeline is::

    HuggingFace Hub
          │
          ▼
    ArchitectureConfig  ◄── from_transformers() / from_diffusers()
          │
          ▼
    Model Module        ◄── reusable Components (Attention, MLP, RMSNorm, RoPE, MoELayer, …)
          │
          ▼
    Task                ◄── CausalLMTask, VisionLanguageTask, VAETask, DenoisingTask, …
          │
          ▼
    ONNX Model          ◄── preprocess_weights() + apply_weights()

The codebase is split into four layers:

* **Components** — ``onnxscript.nn.Module`` building blocks (Attention, MLP,
  DecoderLayer, RoPE, VisionEncoder, MoELayer, ...).
* **Models** — full architectures composed from those components.
* **Tasks** — declare the ONNX graph's I/O contract (inputs, outputs, KV cache),
  e.g. ``CausalLMTask``, ``VisionLanguageTask``, ``VAETask``.
* **Registry** — maps a Hugging Face ``model_type`` string to a model class.

It also layers on **execution-provider-aware optimization** (CUDA / DirectML /
WebGPU graphs with the right fused kernels), an opt-in **static KV cache**, and
coverage that reaches well beyond text generation: MoE, multimodal, encoder-only,
encoder-decoder, speech, audio, vision and diffusion pipelines (hundreds of
Transformers model types plus Diffusers components).

How mbext is organised
----------------------

mbext is a fork of the original ``onnxruntime_genai.models.builder`` and keeps
its command line and conventions (see :doc:`differences_modelbuilder`). A single
entry point, :func:`modelbuilder.builder.create_model`, dispatches on
``config.architectures[0]`` to a ``Model`` subclass in
``modelbuilder/builders/``. Each subclass overrides only the few methods that
differ for its family (``make_layer``, ``make_attention``, ``make_moe``, ...) on
top of a large shared base class that emits the graph imperatively (see
:doc:`design`). Its scope is deliberately narrower — decoder LLMs and MoE for
onnxruntime-genai — and its defining features are:

* **Fast, offline discrepancy tests.** Every architecture ships a tiny,
  random-weight test that converts a one-layer model and compares ONNX vs.
  PyTorch logits within a precision-aware tolerance, so CI stays short and
  deterministic and new architectures get immediate feedback.
* **A** ``--private`` **escape hatch** to convert proprietary architectures that
  live in external files, without modifying the package (see
  :doc:`private_model`).

Side by side
------------

+----------------------------+--------------------------------------------+--------------------------------------------+
|                            | mbext                                      | Mobius                                     |
+============================+============================================+============================================+
| Lineage                    | fork of onnxruntime-genai builder          | separate, newer codebase                   |
+----------------------------+--------------------------------------------+--------------------------------------------+
| Graph construction         | imperative shared base class               | ``onnxscript.nn.Module`` composition       |
+----------------------------+--------------------------------------------+--------------------------------------------+
| Extension model            | subclass a ``Model``, override a few hooks | compose Components → Model → Task, register|
+----------------------------+--------------------------------------------+--------------------------------------------+
| Scope                      | decoder LLMs and MoE                        | LLM, MoE, multimodal, vision, audio,      |
|                            |                                            | encoder-decoder, diffusion                 |
+----------------------------+--------------------------------------------+--------------------------------------------+
| Testing                    | tiny random-weight, fully offline          | unit + integration (downloads models)      |
+----------------------------+--------------------------------------------+--------------------------------------------+
| Private / external models  | built-in ``--private`` option              | via registry / custom components           |
+----------------------------+--------------------------------------------+--------------------------------------------+

Which one to choose to support a new model?
-------------------------------------------

Both are reasonable; the answer depends on the model and the goal.

* **Choose mbext** when the new model is a **decoder-only LLM or MoE** close to a
  family that already exists here. You subclass the nearest builder, override the
  one or two methods that differ, add a dispatch branch, and drop in a tiny
  random-weight test — the offline suite then verifies the conversion in seconds,
  with no downloads. It is also the pragmatic choice for **proprietary models**
  thanks to ``--private``, and for teams already standardised on the
  onnxruntime-genai builder command line.

* **Choose Mobius** when the model needs a modality or pipeline that mbext does
  not target — **vision, audio, encoder-decoder or diffusion** — or when you want
  **execution-provider-tuned graphs** and a **static cache** out of the box. Its
  Component/Model/Task/Registry layering, and its large library of reusable
  components, make novel modalities cheaper to assemble than extending mbext's
  base class would be.

In short: for a new LLM/MoE variant where fast, offline verification matters most,
prefer mbext; for a genuinely new modality or when you need EP-specific graph
optimization, prefer Mobius.

.. note::

   Mobius is evolving quickly. For its current feature set and supported
   architectures, refer to the `Mobius repository
   <https://github.com/onnxruntime/mobius>`_ directly.
