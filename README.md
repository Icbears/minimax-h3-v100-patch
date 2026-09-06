# MiniMax H3 V100 — Custom Node v0.1.4

English | [简体中文](README_zh-CN.md)

Version 0.1.4 fixes `TypeError: _patched_final_forward() takes 5 positional arguments but 8 were given` in ComfyUI 0.34.5 while retaining the standalone v0.1.3 FP16 storage/branch profile and FP32 safety islands. This release has local structural/argument/dtype-flow validation.

## Install / update

1. Fully stop ComfyUI.
2. Remove the previous H3 extension folder from `custom_nodes`.
3. Extract `minimax-h3-v100-v0.1.4.zip` into the active `custom_nodes` directory. The ZIP has one top-level folder:

   ```text
   custom_nodes/minimax-h3-v100-l3-clean/__init__.py
   custom_nodes/minimax-h3-v100-l3-clean/runtime_patch.py
   ```

   In the supplied error report that directory is `C:\Comfyui\custom_nodes`.
4. Restart ComfyUI normally. No new workflow node or `--fp16-unet` flag is needed. Use an official, unmodified ComfyUI `comfy/ldm/minimax/model.py`.
5. Confirm the startup log contains `v0.1.4 runtime profile installed`, then model loading reports `enabled v0.1.4` on the main DiT blocks.

## Precision profile

- Native FP16 weights through ComfyUI loading, prefetch and offload.
- FP16 attention and MLP branches in main DiT blocks; FP32 target/reference-audio attention recomputation.
- Attention output scaling: `/64 → FP16 projection → FP32 ×64`.
- MLP: FP16 `fc1`, FP32 SwiGLU, `/256 → FP16 fc2 → FP32 ×256`.
- FP32 residual, normalization/modulation, condition input, Token Refiner and final heads.
- CUDA compute capability 7.0 restriction, duplicate-extension checks and no lazy `weight.data` conversion remain in place.

The attention and MLP compute functions retain their v0.1.3 AST hashes. The previous v0.1.3 documentation records successful ComfyUI 0.33.2 testing and reduced resident VRAM; that is historical evidence, not a measured result for v0.1.4.

## Acknowledgement and license

Thanks to ComfyUI, MiniMax and [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) for the Custom Node delivery pattern and related mixed-precision work. This is an independent community extension. SPDX: `GPL-3.0-only`; see `LICENSE` and `NOTICE.md`.
