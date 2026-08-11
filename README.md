# MiniMax H3 V100 mixed precision — Custom Node v0.1.2

English | [简体中文](README_zh-CN.md)

Version **0.1.2** delivers the tested MiniMax H3 V100 mixed-precision profile as a ComfyUI Custom Node. This is now the recommended installation method: it does not rewrite `comfy/ldm/minimax/model.py`, does not require a modified launch command, and can be removed by deleting one folder and restarting ComfyUI.

The Custom Node mode supports the updated MiniMax H3 audio carry/sampler structure in **ComfyUI 0.31.1**. The user has reported successful V100 operation after testing this runtime build.

The previous source-modifying installers are still available under [`legacy_patcher/`](legacy_patcher/) for recovery, development, and users who explicitly need them.

## Recommended Custom Node installation

### Before installing

Use a clean, unpatched ComfyUI `comfy/ldm/minimax/model.py`. If an earlier release of this project modified that file, first stop ComfyUI and restore it with the matching script:

```bat
legacy_patcher\restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

The runtime extension intentionally refuses to stack on top of the old source patch or another H3 monkey patch.

### Install

1. Stop ComfyUI.
2. Download or clone this repository.
3. Put the complete repository folder under `ComfyUI/custom_nodes/`. The important layout is:

   ```text
   ComfyUI/
   └─ custom_nodes/
      └─ minimax-h3-v100-patch/
         ├─ __init__.py
         ├─ runtime_patch.py
         ├─ README.md
         └─ legacy_patcher/
   ```

4. Start ComfyUI normally. Do **not** add `--fp16-unet`, `--force-fp16`, or another precision flag for this profile.
5. Confirm that the startup log contains:

   ```text
   [MiniMax H3 V100] v0.1.2 runtime profile installed: FP32 base + FP16 QKV/text-video attention + FP32 audio attention
   [MiniMax H3 V100] no --fp16-unet flag is required; source files remain untouched
   ```

The first H3 model load also reports how many main DiT blocks were enabled. The extension adds no workflow node; existing MiniMax H3 workflows continue to be used normally.

### Update or uninstall

- Update: stop ComfyUI, replace this repository folder with the new version, and restart.
- Uninstall: delete this repository folder from `custom_nodes` and restart. No ComfyUI source backup needs to be restored when only the Custom Node mode was used.

## Precision split

The 50 main DiT blocks use the established FP32-base profile:

- FP16: resident fused-QKV weights, fused QKV projection, and text/video attention queries.
- FP32: target/reference-audio attention queries, Q/K RMSNorm, RoPE, attention output projection, MLP, AdaLN, residual accumulation, and final output heads.
- Source compute path: the two token-refiner blocks.

Audio FP32 attention receives Q/K/V values produced by the FP16 QKV projection, while the audio attention operation itself is FP32. The profile activates only for CUDA FP32 main-block input during inference. CPU, BF16, global-FP16, and training calls delegate to the source implementation.

## Runtime safety

- Checks the current `Attention`, `DiTBlock`, `MLP`, `MiniMaxH3Model`, and `PackedLayout` structure before installation.
- Refuses source-level V100 patches and other extensions that already own critical H3 methods.
- Rechecks for extensions loaded later and keeps new model instances unmodified if a conflict appears.
- Installs idempotently.
- Missing or invalid audio row metadata fails closed to full FP32 attention for that call.
- QKV weights outside FP32/FP16 are not force-converted; that block returns to the source path.
- Does not open, replace, rename, or write any ComfyUI source file.

This runtime build targets the ComfyUI 0.31.1 H3 structure containing the updated audio carry/sampler path and `PackedLayout.segments`. A future ComfyUI internal refactor may cause an intentional automatic disable instead of an unsafe partial patch.

## Performance evidence

Earlier V100 test case: 0.2 MP, 5 s, 24 fps. Environment: Windows 10, NVIDIA Tesla V100 32 GB, 150 W power limit; the baseline already included TE-Speed.

| Profile | Time | Change from baseline | Result |
|---|---:|---:|---|
| TE-Speed baseline | 317 s | — | Passed |
| **MiniMax H3 V100 mixed-precision profile** | **206 s** | **35.0% faster / 1.54x** | **Passed** |

These figures demonstrate the earlier core precision profile and are not a universal timing guarantee for every ComfyUI workflow. In the later audio-safe build, resident FP16 QKV weights provided an additional reported improvement of approximately **5%–10%** over recasting those weights during inference. Actual results depend on workload, backend, offload behavior, power limit, and build.

For the first A/B test, keep the seed, prompt, model, sampler, steps, resolution, frame count, and attention backend identical. Disable unrelated TeaCache, SageAttention, compilation, global FP16, and FP16-accumulation changes; discard one warm-up run and record at least three measured runs. Stop at the first black frame, NaN, audio defect, or error.

## Legacy source patcher

The pre-0.1.2 source-modifying package is preserved under [`legacy_patcher/`](legacy_patcher/). It contains:

- guarded origin and TE-Speed BAT/Python installers;
- guarded restore scripts and exact adjacent-backup handling;
- origin/TE structure detection, AST validation, idempotence, and SHA-256 checks;
- complete V100 source snapshots for development and manual porting;
- its own manifest and provenance notice.

The Custom Node is recommended for normal use. Use the legacy patcher only when you intentionally need a modified `model.py`, a TE `block_loop` source conversion, recovery from an earlier patch, or inspectable full source snapshots.

Windows entry points:

```bat
legacy_patcher\patch_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\patch_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_te_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
legacy_patcher\restore_h3_origin_v100.bat "D:\ComfyUI\comfy\ldm\minimax\model.py"
```

Python entry points:

```text
python legacy_patcher/patch_te_v100.py /path/to/ComfyUI
python legacy_patcher/patch_h3_origin_v100.py /path/to/ComfyUI
```

Development source snapshots:

- [`legacy_patcher/sources/origin_v100/model.py`](legacy_patcher/sources/origin_v100/model.py): official/origin loop plus the audio-safe V100 profile.
- [`legacy_patcher/sources/te_v100/model.py`](legacy_patcher/sources/te_v100/model.py): TE-Speed block-loop hooks plus the same profile.

The supported clean origin source is derived from the ComfyUI implementation containing audio fix [`93cb5edb`](https://github.com/Comfy-Org/ComfyUI/commit/93cb5edb) and has SHA-256 `1c9828ec3d38ac01398e45b1edf8d7db38fcc8148c5eb3ba8fb92b762147d0ce`. See [`legacy_patcher/NOTICE.md`](legacy_patcher/NOTICE.md) for the original patcher evidence and [`NOTICE.md`](NOTICE.md) for the v0.1.2 runtime boundary.

## Acknowledgements

Special thanks to [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix). That project demonstrated that MiniMax H3 acceleration could be delivered cleanly through a ComfyUI Custom Node and inspired the v0.1.2 move away from persistent source edits.

This project keeps its existing, independently tested **FP32 ocean + FP16 hotspot islands** precision strategy; the acknowledgement is specifically for the Custom Node delivery approach. Thanks also to ComfyUI, MiniMax, and the TE-Speed contributors whose work forms the surrounding ecosystem.

## Scope and license

This is a community mixed-precision extension, not an official MiniMax, TE-Speed, ComfyUI, or acknowledged-project release. SPDX license identifier: `GPL-3.0-only`; see [`LICENSE`](LICENSE). The acknowledged project is separately licensed by its author.
