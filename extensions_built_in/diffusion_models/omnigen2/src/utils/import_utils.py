# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.util
import sys

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

def _is_package_available(pkg_name: str):
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        try:
            pkg_version = importlib_metadata.version(pkg_name)
        except (ImportError, importlib_metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version

_triton_available, _triton_version = _is_package_available("triton")
_flash_attn_available, _flash_attn_version = _is_package_available("flash_attn")

def is_triton_available():
    return _triton_available

def is_flash_attn_available():
    return _flash_attn_available