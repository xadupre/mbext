Differences with Mobius (onnxruntime/mobius)
============================================

`Mobius <https://github.com/onnxruntime/mobius>`_ is a newer model builder for
onnxruntime-genai. Like mbext, it constructs ONNX graphs declaratively (using
`onnxscript <https://github.com/microsoft/onnxscript>`_) and applies Hugging Face
or custom weights, rather than tracing an existing PyTorch graph. Both projects
target the same runtime (onnxruntime-genai) and the same kinds of models (LLMs,
MoE, multimodal, ...).

Where they differ
-----------------

**Lineage.** mbext is a fork of the original ``onnxruntime_genai.models.builder``
and keeps its command line and conventions (see
:doc:`differences_modelbuilder`). Mobius is a separate, more recent codebase.

**Testing focus.** mbext's defining feature is its short CI and fast,
random-weight discrepancy tests that run fully offline (see :doc:`design`). The
project is organised around keeping that suite fast and deterministic so new
architectures get immediate feedback.

**Extensibility for private models.** mbext ships a ``--private`` option to
convert proprietary architectures that live in external files, without modifying
the package (see :doc:`private_model`).

Which one should I use?
-----------------------

If you are already using the classic onnxruntime-genai model builder and want a
drop-in replacement with more architectures, faster tests and a private-model
escape hatch, mbext is a natural fit. If you are starting fresh and want the
officially maintained onnxruntime pipeline, evaluate Mobius as well; the two
share enough concepts that experience transfers between them.

.. note::

   Mobius is evolving quickly. For its current feature set and supported
   architectures, refer to the `Mobius repository
   <https://github.com/onnxruntime/mobius>`_ directly.
