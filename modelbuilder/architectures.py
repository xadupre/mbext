# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Introspection helpers describing the architectures supported by mbext.

The list of supported Hugging Face architectures is not maintained by hand.
Instead, it is extracted from the dispatch chain of
:func:`modelbuilder.builder.create_model` by parsing its source code. This keeps
the documentation (see ``docs/architectures.rst``) automatically in sync with
the code: a new ``elif config.architectures[0] == "..."`` branch shows up in the
list without any extra bookkeeping.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SupportedArchitecture:
    """Description of a single supported architecture.

    Attributes:
        name: The Hugging Face architecture name, i.e. the value found in
            ``config.architectures[0]`` (for example ``"LlamaForCausalLM"``).
        builder_module: Dotted path of the module defining the builder, for
            example ``"modelbuilder.builders.llama"``. May be empty if it could
            not be determined statically.
        builder_class: Name of the builder class handling the architecture, for
            example ``"LlamaModel"``. May be empty if it could not be determined
            statically.
    """

    name: str
    builder_module: str = ""
    builder_class: str = ""


@dataclass
class _Branch:
    names: list[str] = field(default_factory=list)
    module: str = ""
    cls: str = ""


def _architecture_names_in_test(test: ast.AST) -> list[str]:
    """Return the architecture string literals compared in a branch test.

    Handles both ``config.architectures[0] == "X"`` and
    ``config.architectures[0] in ("X", "Y")`` comparisons (possibly combined
    with ``and``/``or`` in a boolean expression).
    """

    names: list[str] = []
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        # Both ``config.architectures[0] == "X"`` and
        # ``config.architectures[0] in ("X", "Y")`` keep ``config.architectures``
        # on the left and the string literals on the right.
        if not _references_architecture(node.left):
            continue
        for comparator in node.comparators:
            names.extend(_string_constants(comparator))
    return names


def _references_architecture(node: ast.AST) -> bool:
    """Return True if the node is ``config.architectures[0]``."""
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "architectures"


def _string_constants(node: ast.AST) -> list[str]:
    """Return the string literals contained in a constant, tuple or list."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_string_constants(elt))
        return out
    return []


def _builder_in_body(body: list[ast.stmt]) -> tuple[str, str]:
    """Return ``(module, class)`` of the builder imported/used in a branch.

    Walks the whole branch body (including nested ``if``/``else`` blocks) so
    branches that select the builder conditionally are still resolved.
    """
    module = ""
    cls = ""
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.ImportFrom) and node.module and "builders" in node.module:
                mod = node.module
                if node.level and node.level > 0:
                    mod = "modelbuilder." + mod
                if not module:
                    module = mod
                    if node.names:
                        cls = node.names[0].name
    return module, cls


def list_supported_architectures() -> list[SupportedArchitecture]:
    """List the Hugging Face architectures supported by :func:`create_model`.

    The list is produced by statically parsing the dispatch chain of
    :func:`modelbuilder.builder.create_model`, so it always reflects the current
    state of the code. Results are sorted alphabetically by architecture name.

    The parsing reads the ``builder.py`` source file directly (without importing
    it), so this helper stays lightweight and usable in environments where heavy
    dependencies such as ``torch`` are not installed (for example when building
    the documentation).
    """

    builder_path = os.path.join(os.path.dirname(__file__), "builder.py")
    with open(builder_path, encoding="utf-8") as f:
        source = f.read()
    module_tree = ast.parse(source)
    func = None
    for node in module_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "create_model":
            func = node
            break
    if func is None:
        raise RuntimeError("Could not find create_model in builder.py")

    branches: list[_Branch] = []

    def visit_if(node: ast.If) -> None:
        names = _architecture_names_in_test(node.test)
        if names:
            module, cls = _builder_in_body(node.body)
            branches.append(_Branch(names=names, module=module, cls=cls))
        for stmt in node.orelse:
            if isinstance(stmt, ast.If):
                visit_if(stmt)

    for stmt in func.body:
        if isinstance(stmt, ast.If):
            visit_if(stmt)

    seen: dict[str, SupportedArchitecture] = {}
    for branch in branches:
        for name in branch.names:
            # Keep the first builder seen for a given architecture name.
            seen.setdefault(name, SupportedArchitecture(name, branch.module, branch.cls))

    return sorted(seen.values(), key=lambda a: a.name.lower())
