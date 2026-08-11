# Provenance and validation notice

## Release

- Package version: `0.1.2`
- Default delivery: ComfyUI Custom Node loaded at runtime
- Supported target used for structural and runtime validation: ComfyUI `0.31.1` MiniMax H3 with the updated audio carry/sampler path
- Supported clean origin `model.py` SHA-256: `1c9828ec3d38ac01398e45b1edf8d7db38fcc8148c5eb3ba8fb92b762147d0ce`
- Relevant upstream audio fix: ComfyUI commit `93cb5edb`

## Precision-profile provenance

Version 0.1.2 preserves the existing audio-safe V100 profile:

- FP16 resident QKV weights, QKV projection, and text/video attention in the main DiT blocks.
- FP32 target/reference-audio attention, Q/K RMSNorm, RoPE, output projection, MLP, AdaLN, residual accumulation, and final heads.
- Source behavior for the two token-refiner blocks.

The Custom Node changes only how this established profile is delivered. Import-time wrappers replace persistent edits to `comfy/ldm/minimax/model.py`.

## Runtime implementation

- `MiniMaxH3Model.__init__` enables the profile only on the main DiT blocks.
- `MiniMaxH3Model._forward` and `PackedLayout.__init__` expose target/reference-audio row ranges through task-local runtime context.
- `Attention.forward` performs the FP16 QKV/text-video work and recomputes audio query rows with FP32 attention.
- `DiTBlock.forward`, `MLP.forward`, AdaLN, residual code, output projections, token refiners, and final heads are not replaced.
- Method-ownership and structural checks refuse incompatible or stacked H3 runtime patches.
- Missing/invalid audio row metadata fails closed to full FP32 attention for that call.

## Validation completed on 2026-08-11

- The user reported successful ComfyUI 0.31.1 V100 operation with normal output after testing the new Custom Node delivery.
- Python AST parsing passed for the package, runtime implementation, and tests.
- Dependency-free simulated-loader tests cover installation, version reporting, idempotence, main-block-only activation, token-refiner exclusion, audio-range capture/reset, invalid-range fail-closed behavior, early/late conflict refusal, and absence of source-file write APIs.
- The supported clean origin text was reconstructed in memory from the verified bundled output; its SHA-256 matched the expected clean hash exactly.
- No ComfyUI source file is modified by the Custom Node.

## Legacy patcher

The earlier guarded source patchers, restore scripts, source snapshots, their original manifest, and their detailed provenance notice are retained under `legacy_patcher/`. They remain available for recovery, development, and users who explicitly need source-modifying installation, but the Custom Node is the recommended path for normal use.

## Acknowledgement

Thanks to [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) for demonstrating a practical ComfyUI Custom Node delivery pattern for MiniMax H3 acceleration. That project inspired the move away from persistent source edits. Version 0.1.2 retains this project's independently tested FP32-base precision boundary; this acknowledgement concerns the deployment approach.

## Licensing

SPDX-License-Identifier: GPL-3.0-only. See `LICENSE`. The acknowledged project is separately licensed by its author.
