# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Sphinx extension exposing the ``supported-architectures`` directive.

The directive renders the list of Hugging Face architectures supported by
:func:`modelbuilder.builder.create_model` as a table. The list is computed at
build time by :func:`modelbuilder.architectures.list_supported_architectures`,
so the documentation page is always in sync with the code.
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive

from modelbuilder.architectures import list_supported_architectures


class SupportedArchitecturesDirective(Directive):
    has_content = False

    def run(self):
        archs = list_supported_architectures()

        table = nodes.table()
        tgroup = nodes.tgroup(cols=2)
        table += tgroup
        for width in (50, 50):
            tgroup += nodes.colspec(colwidth=width)

        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        for label in ("Architecture", "Builder"):
            entry = nodes.entry()
            entry += nodes.paragraph(text=label)
            header_row += entry
        thead += header_row

        tbody = nodes.tbody()
        tgroup += tbody
        for arch in archs:
            row = nodes.row()

            name_entry = nodes.entry()
            name_para = nodes.paragraph()
            name_para += nodes.literal(text=arch.name)
            name_entry += name_para
            row += name_entry

            builder_entry = nodes.entry()
            builder_para = nodes.paragraph()
            if arch.builder_module and arch.builder_class:
                builder_para += nodes.literal(text=f"{arch.builder_module}.{arch.builder_class}")
            builder_entry += builder_para
            row += builder_entry

            tbody += row

        count = nodes.paragraph()
        count += nodes.Text(f"{len(archs)} architectures are currently supported.")

        return [count, table]


def setup(app):
    app.add_directive("supported-architectures", SupportedArchitecturesDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
