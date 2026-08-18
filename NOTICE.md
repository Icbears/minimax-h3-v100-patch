# Provenance and validation notice

## Release

- Package version: `0.1.3`
- Default delivery: standalone ComfyUI Custom Node loaded at runtime
- Target GPU: NVIDIA Tesla V100 / CUDA compute capability 7.0
- Current reported compatibility: ComfyUI `0.33.2`
- Profile: successful L3 native-FP16 storage/branch profile with FP32 safety islands

## Precision-profile provenance

Version 0.1.3 promotes the successful, debug-free L3 profile to the formal release:

- Native FP16 weights through ComfyUI loading, prefetch and offload.
- FP16 QKV, Q/K RMSNorm, RoPE and text/video attention in the main DiT blocks.
- FP32 target/reference-audio attention recomputation.
- Power-of-two scaling around FP16 attention output projection and MLP `fc2`.
- FP32 main residual, Block Norm, AdaLN/modulation, condition input, Token Refiner, SwiGLU intermediates, final AdaLN and video/audio heads.

Full-FP16 final heads remain excluded. The implementation does not mutate `weight.data`, retain a complete FP32 weight mirror across blocks, or convert full weights inside `forward`.

## Runtime implementation

- Registers FP16 as a supported MiniMax H3 inference dtype before model construction so ComfyUI owns the FP16 weight lifecycle.
- Enables the profile only on native-FP16 main DiT blocks; the Token Refiner remains outside the accelerated block profile.
- Preserves target/reference-audio row ranges through task-local runtime context.
- Uses `/64 → FP16 out_proj → FP32 ×64` and `/256 → FP16 fc2 → FP32 ×256`.
- Forces `condition_proj` input and both final output heads through FP32 safety paths.
- Refuses unsupported GPUs, unfamiliar ComfyUI H3 structures, source patches, pre-existing FP16 dtype providers and late runtime conflicts.
- Contains no per-forward allocator telemetry, OOM diagnostic interception, cache clearing, CUDA synchronization or peak-counter reset.
- Does not modify any ComfyUI source file.

## Validation completed on 2026-08-18

- The user reported that the L3 profile successfully reduced inference-period resident VRAM while keeping inference performance unchanged.
- The user reported successful operation with ComfyUI 0.33.2.
- Fourteen dependency-free tests cover version reporting, V100 restriction, native-FP16 registration, main-block/Token-Refiner isolation, dtype flow, FP32 audio/final safety, power-of-two scaling, idempotence, early/late conflict refusal, no lazy weight mutation, no source writes and no allocator debug telemetry.
- The promoted release compute-kernel AST hashes match the successful L3 profile.

Target-GPU results remain workload-specific; users should hold model, seed, prompt, sampler, steps, dimensions, attention backend, offload mode and power limit constant when comparing releases.

## Compatibility boundary

This is the standalone v0.1.3 runtime profile. It must not be stacked with TE-Speed or another H3 runtime/dtype patch. A separately reviewed adapter is required for combined execution.

## Legacy patcher

The earlier guarded source patchers, restore scripts, source snapshots, their original manifest and detailed provenance notice remain under `legacy_patcher/`. They are retained for recovery and development and are not the recommended v0.1.3 installation path.

## Acknowledgement

Thanks to [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) for demonstrating a practical ComfyUI Custom Node delivery pattern and the native-FP16/FP32-residual design with power-of-two projection scaling. This is a community extension, not an official MiniMax, ComfyUI, NVIDIA, PyTorch, TE-Speed or acknowledged-project release.

## Licensing

SPDX-License-Identifier: GPL-3.0-only. See `LICENSE`. The acknowledged project is separately licensed by its author.
