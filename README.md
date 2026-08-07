# MiniMax H3 V100 mixed-precision patchers

English | [简体中文](README_zh-CN.md)

Two version-gated installers for the tested MiniMax H3 V100 mixed-precision profile:

- `patch_te_v100.bat` installs the TE-Speed version. On a supported clean origin file, it first installs the verified TE-Speed `block_loop` hooks and then applies the V100 patch.
- `patch_h3_origin_v100.bat` patches the official/origin H3 `model.py` without those TE hooks.
- `restore_te_v100.bat` restores the exact pre-patch TE backup.
- `restore_h3_origin_v100.bat` restores the exact pre-patch origin backup.

Do **not** run both patchers on the same file. The TE installer can promote a supported clean origin file to the TE layout; the origin installer still accepts only the official origin layout.

## Measured result

Test case supplied by the V100 tester: 0.2 MP, 5 s, 24 fps.

Test environment: Windows 10, NVIDIA Tesla V100 32 GB, with the GPU power limit set to 150 W. The baseline already included the TE-Speed optimization. Better cooling and a higher power limit may provide additional performance; operation at 300 W is expected to be faster, but has not yet been benchmarked and is not included in the results below.

| Profile | Time | Change from baseline | Result |
|---|---:|---:|---|
| TE-Speed baseline | 317 s | — | Passed |
| **MiniMax H3 V100 mixed-precision patch** | **206 s** | **35.0% faster / 1.54x** | **Passed** |

These patchers install the tested **MiniMax H3 V100 mixed-precision patch**. Results are workload-, backend- and build-dependent; 1.54x is the measured example, not a universal guarantee.

## Precision split

The 50 main DiT blocks use:

- FP16: fused QKV projection and the optimized attention kernel.
- FP32: Q/K RMSNorm, RoPE, attention output projection, MLP, AdaLN, residual accumulation and final output heads.
- Source compute dtype: the two token-refiner blocks.

The patch activates only when the incoming main-block tensor is CUDA FP32. BF16 and already-FP16 paths retain the source behavior.

## Bundled source files

Complete V100-accelerated `model.py` sources are included for developers who want to inspect, adapt or integrate the changes into their own ComfyUI builds:

- [`sources/origin_v100/model.py`](sources/origin_v100/model.py): official/origin MiniMax H3 structure plus the tested V100 Plan 2 mixed-precision acceleration; no TE-Speed `block_loop` hook.
- [`sources/te_v100/model.py`](sources/te_v100/model.py): TE-Speed `block_loop` hooks plus the same tested V100 Plan 2 acceleration.

Both sources are derived from the ComfyUI MiniMax H3 implementation introduced by commit [`57500fc`](https://github.com/Comfy-Org/ComfyUI/commit/57500fc), whose clean origin `model.py` has SHA-256 `882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024`.

For a matching ComfyUI build, either file can be used as a drop-in `comfy/ldm/minimax/model.py` after making a backup. For newer, older or locally modified builds, use these files as reference sources and port the Attention precision changes and, when desired, the TE block-loop hooks into the local implementation instead of blindly overwriting it. Stop ComfyUI before replacing the file and validate the resulting Python syntax and model output before production use. These source copies are provided for manual development; the guarded BAT/Python installers remain the safer option for the supported build.

## One-click Windows usage

1. Stop ComfyUI.
2. Choose the BAT matching the active `model.py`:
   - TE-Speed version required (clean origin or already TE-hooked): `patch_te_v100.bat`
   - Official/origin H3 file: `patch_h3_origin_v100.bat`
3. Double-click the selected BAT. This customized build defaults to `C:\Users\Administrator\ComfyUI-Installs\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py`; another `model.py` can still be dragged onto the BAT.
4. Check that the console reports `Patched SHA-256`, then restart ComfyUI.

To return to the pre-patch file, stop ComfyUI and run the matching restore BAT. If TE installation started from origin, `restore_te_v100.bat` restores that exact origin file; if it started from an existing TE file, it restores that exact TE file.

The BAT launchers try, in order: `COMFYUI_PYTHON`, a nearby portable `python_embeded\python.exe`, the Windows `py -3` launcher, then `python` from `PATH`.

Command-line examples:

```bat
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --check
patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --revert

patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --check
patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py" --revert

restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

The scripts can also auto-locate common ComfyUI/portable layouts when launched from inside the installation. Use `--comfy-ui <root>` or `--model-file <file>` when more than one installation exists.

## Backups and safety gates

Before the first write, the installers make an exact adjacent backup:

- TE variant: `model.py.v100_te.bak` (the exact pre-install file, which may be origin or TE)
- Origin variant: `model.py.v100_origin.bak`

Safety behavior:

- Exact TE/origin layout detection before patching; origin-to-TE promotion runs only when both verified TE conversion anchors match.
- Exact supported `Attention.forward` anchor required.
- Refuses partial TE hooks, partial V100 patches and wrong-target patchers.
- Refuses to overwrite a differing/stale backup.
- Writes through a temporary file followed by an atomic replacement.
- Python AST syntax validation before and after the write.
- SHA-256 printed for the input, patched output and restored output.
- Repeated patch calls are idempotent.
- `--revert` works only when the active file is recognized as V100-patched and its matching clean backup is valid.
- The dedicated restore BATs call that same guarded `--revert` path; they do not perform an unchecked file copy.

## Benchmarking guidance

- Keep the H3 compute/residual stream FP32; do not combine the first comparison with global `--force-fp16` or BF16.
- Keep seed, prompt, sampler, steps, resolution, frames, checkpoint and attention backend identical.
- Disable unrelated TeaCache, SageAttention, compilation and FP16-accumulation changes for the initial A/B test.
- Discard one warm-up run and time at least three measured runs.
- Record total time, seconds per step, peak VRAM, video validity, audio validity and the first warning/error.

## Python entry points

The BAT files are thin Windows launchers. The dependency-free Python installers work directly on Windows or Linux:

```text
python patch_te_v100.py /path/to/ComfyUI
python patch_h3_origin_v100.py /path/to/ComfyUI
```

Python 3.8 or newer is recommended.

## Scope and attribution

This is a community mixed-precision patch, not an official MiniMax, TE-Speed or ComfyUI release. It modifies the upstream ComfyUI MiniMax H3 model implementation and preserves the TE hooks when the TE-specific installer is used. See `NOTICE.md` for source hashes and validation provenance.

SPDX license identifier: `GPL-3.0-only`, matching the upstream ComfyUI license used by the modified implementation. See `LICENSE`.
