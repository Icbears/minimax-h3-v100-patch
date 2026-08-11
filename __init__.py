"""ComfyUI loader entry point for the MiniMax H3 V100 mixed-precision profile."""

__version__ = "0.1.2"

from .runtime_patch import PATCH_STATUS, install_patch


# This extension changes H3 internals at import time and intentionally exposes no
# workflow node. Keeping these mappings empty makes it a drop-in custom_nodes
# extension without adding UI clutter.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PATCH_STATUS",
    "__version__",
    "install_patch",
]
