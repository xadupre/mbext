Supported architectures
=======================

The table below lists every Hugging Face architecture that
:func:`modelbuilder.builder.create_model` can convert, together with the builder
class that handles it. It is generated automatically at documentation build time
by parsing the dispatch chain of ``create_model`` (see
:func:`modelbuilder.architectures.list_supported_architectures`), so it is always
in sync with the code.

The architecture name is the value of ``config.architectures[0]`` in the Hugging
Face ``config.json``. To convert a model of a given family, point ``-m`` or
``-i`` at a checkpoint whose config declares that architecture.

.. supported-architectures::

Qwen2.5-Omni video inputs
------------------------

Export with ``multimodal=True`` to produce the shared image/video vision
encoder, audio encoder, embedding model, and thinker decoder. For direct
ONNX Runtime inference with multimodal temporal/spatial positions, also set
``use_3d_position_ids=True`` (CLI:
``--extra_options multimodal=true use_3d_position_ids=true``).
The default decoder still accepts 2-D position IDs for existing callers.

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

This video path uses direct ONNX Runtime sessions. The existing ``phi3v``
ORT-GenAI configuration cannot supply vision rotary embeddings, frame/window
IDs, or Omni's video timing; automatic video processing through ORT-GenAI
is not supported. No unsupported video fields are added to its configuration.

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
