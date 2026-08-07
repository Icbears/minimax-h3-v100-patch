# Provenance and validation notice

## Supplied inputs

- Official/origin H3 backup: `model.py.te_speed.bak`
  - Size: 33,259 bytes
  - SHA-256: `882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024`
- TE-hooked H3 model: `model.py`
  - Size: 35,173 bytes
  - SHA-256: `1f95147e69215d7b015e553ef20be20a766291feded683cd953d403c433b7a93`
- TE hook reference installer: `patch_model.py`
  - Size: 7,727 bytes
  - SHA-256: `fae0c475ae1a90ec6ae3d606fa8bed4e87d6c90b4b0d4fbeb587f02ec0897ec5`
- Mixed-precision Qwen reference: `model (1).py`
  - Size: 27,886 bytes
  - SHA-256: `8516c792f42bac6f6d339580808594d0d9b6fc176d4a188691b43d399868b324`

The supplied origin and TE H3 files differ only by the TE-Speed `_run_blocks(start, end)` method and `("block_loop", 0)` forward hook. Their Attention implementation is identical before this V100 patch.

## Bundled V100 source snapshots

- `sources/origin_v100/model.py` contains the supported official/origin structure with the tested V100 Plan 2 precision patch applied.
- `sources/te_v100/model.py` contains the verified TE-Speed hooks with the same V100 Plan 2 precision patch applied.

The bundled files reproduce the installer-generated output hashes recorded in `MANIFEST.json`: `ff86e416...` for origin and `24c3a267...` for TE promoted from origin. Python AST parsing, variant detection and V100 patch-marker validation are run against both bundled files.

## Precision selection evidence

The published patch is the V100-tested Plan 2 profile:

- Baseline: 317 s
- Plan 1: 260 s, passed
- Plan 2: 206 s, passed
- Plan 3: failed
- Plan 4: failed

Test case as reported by the tester: 0.2 MP, 5 s, 24 fps.

## Local installer validation

Validated on isolated copies of both supplied files:

- Clean source detection and `--check`.
- Patch plus Python AST parse.
- Patched origin and TE `Attention.forward` and `MiniMaxH3Model.__init__` ASTs are identical to the V100-tested Plan 2 file.
- Repeated invocation produces no byte changes.
- The origin patcher rejects TE input; the TE patcher safely promotes the verified origin input before applying the V100 patch.
- `--revert` restores the exact pre-install SHA-256, including origin input promoted by the TE patcher.
- Both patch BAT launchers, both dedicated restore BAT launchers and drag-and-drop positional path parsing were exercised.

The development machine did not contain a V100, so the performance claim comes from the tester's V100 run rather than a local rerun.
