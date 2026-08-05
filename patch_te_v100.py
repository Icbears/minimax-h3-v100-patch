#!/usr/bin/env python3
"""Patch the TE-Speed-hooked MiniMax H3 model for NVIDIA V100."""

# SPDX-License-Identifier: GPL-3.0-only

from _patch_core import main


if __name__ == "__main__":
    raise SystemExit(main("te"))
