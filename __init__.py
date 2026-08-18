"""ComfyUI entry point for the MiniMax H3 V100 v0.1.3 profile."""

__version__ = "0.1.3"

from .runtime_patch import PATCH_STATUS, install_patch


# Import-time runtime extension: existing H3 workflows need no extra UI node.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PATCH_STATUS",
    "__version__",
    "install_patch",
]
