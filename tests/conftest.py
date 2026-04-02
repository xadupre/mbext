# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import sys
import os

# Add the modelbuilder directory to sys.path so that `builder` and `builders` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modelbuilder"))
