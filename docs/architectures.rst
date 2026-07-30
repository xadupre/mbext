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
