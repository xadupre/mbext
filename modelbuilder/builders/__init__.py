# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# -------------------------------------------------------------------------
# Copyright (C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
__all__ = ["Model"]


def __getattr__(name):
    if name == "Model":
        from .base import Model

        globals()[name] = Model
        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
