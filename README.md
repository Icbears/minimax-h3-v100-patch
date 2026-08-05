# MiniMax H3 V100 mixed-precision patchers

Two version-gated installers for the tested MiniMax H3 V100 mixed-precision profile:

- `patch_te_v100.bat` patches a `model.py` that already contains the TE-Speed `block_loop` hooks.
- `patch_h3_origin_v100.bat` patches the official/origin H3 `model.py` without those TE hooks.
- `restore_te_v100.bat` restores the exact pre-patch TE backup.
- `restore_h3_origin_v100.bat` restores the exact pre-patch origin backup.

Do **not** run both patchers on the same file. Each patcher detects the target layout and refuses the wrong variant before writing anything.

## Measured result

Test case supplied by the V100 tester: 0.2 MP, 5 s, 24 fps.

| Profile | Time | Change from baseline | Result |
|---|---:|---:|---|
| Original baseline | 317 s | — | Passed |
| Plan 1: FP16 attention kernel only | 260 s | 18.0% faster / 1.22x | Passed |
| **Plan 2: FP16 QKV + attention, FP32 Q/K norm/RoPE/residual** | **206 s** | **35.0% faster / 1.54x** | **Passed** |
| Plan 3: alternating FP16 attention-output/MLP GEMMs | — | — | Failed |
| Plan 4: all-block FP16 attention-output/MLP GEMMs | — | — | Failed |

These patchers install the successful **Plan 2** profile. Results are workload-, backend- and build-dependent; 1.54x is the measured example, not a universal guarantee.

## Precision split

The 50 main DiT blocks use:

- FP16: fused QKV projection and the optimized attention kernel.
- FP32: Q/K RMSNorm, RoPE, attention output projection, MLP, AdaLN, residual accumulation and final output heads.
- Source compute dtype: the two token-refiner blocks.

The patch activates only when the incoming main-block tensor is CUDA FP32. BF16 and already-FP16 paths retain the source behavior.

## One-click Windows usage

1. Stop ComfyUI.
2. Choose the BAT matching the active `model.py`:
   - TE-Speed hooks already installed: `patch_te_v100.bat`
   - Official/origin H3 file: `patch_h3_origin_v100.bat`
3. Drag `ComfyUI\comfy\ldm\minimax\model.py` onto the selected BAT file.
4. Check that the console reports `Patched SHA-256`, then restart ComfyUI.

To return to the pre-patch file, stop ComfyUI and drag the active `model.py` onto the matching restore BAT. The restore script validates the active variant, V100 patch markers and matching clean backup before it writes anything.

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

- TE variant: `model.py.v100_te.bak`
- Origin variant: `model.py.v100_origin.bak`

Safety behavior:

- Exact TE/origin layout detection before patching.
- Exact supported `Attention.forward` anchor required.
- Refuses partial TE hooks, partial V100 patches and wrong-target patchers.
- Refuses to overwrite a differing/stale backup.
- Writes through a temporary file followed by an atomic replacement.
- Python AST syntax validation before and after the write.
- SHA-256 printed for the input, patched output and restored output.
- Repeated patch calls are idempotent.
- `--revert` works only when the active file is recognized as V100-patched and its matching clean backup is valid.
- The dedicated restore BATs call that same guarded `--revert` path; they do not perform an unchecked file copy.

## Known supplied builds

| Variant | Supplied input SHA-256 | Expected patched SHA-256 |
|---|---|---|
| Official/origin H3 | `882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024` | `ff86e416f7fb25d9cb68ceda6fb3e63f95e919db51c019a5a570171bbcd28b78` |
| TE-hooked H3 | `1f95147e69215d7b015e553ef20be20a766291feded683cd953d403c433b7a93` | `53167a18ad0872d26a9e3502b9a6bc7c135bb10034c60f68e4ad9e93a93d55c6` |

An unknown whole-file hash may still be accepted only when all exact structural and Attention anchors match. A changed Attention implementation is refused and must be reviewed manually.

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
