# Provenance and validation notice

Version: 0.1.4. Date: 2026-09-06. Target: CUDA compute capability 7.0 (Volta/V100).

This standalone Custom Node extends the v0.1.3 native-FP16 L3 release. Attention and MLP compute-function AST hashes are unchanged. FinalLayer now delegates upstream after promoting x and t_emb to FP32; all trailing positional and keyword arguments survive. This retains upstream modulation/PDD semantics. The DiT wrapper also supports the upstream optional attention override. Unknown parameters on reimplemented methods disable installation before dtype registration.

Local validation: 23 dependency-free tests pass, including execution of real v0.34.5/PDD FinalLayer source bodies with trace tensors. PDD arithmetic is replaced by a spy; argument routing, schedule interval selection, per-stream shifts and FP32 flow are checked. No real PyTorch tensor, V100 inference, image/audio quality, memory or speed result is claimed for v0.1.4. Earlier 0.1.3 runtime results are historical only.

Upstream source fixtures are unchanged ComfyUI model.py snapshots retrieved on 2026-09-06. They are test data, never installed over ComfyUI. Their URLs and SHA-256 digests are recorded in tests/fixtures/PROVENANCE.json. The current-master snapshot was verified against fixed commit 15eb748b3ec5f8a0a2d470b7fb280e2d7579f916.

The v0.1.4 distribution excludes TE adapters, TE-Speed and legacy source patchers. It must be installed as the only H3 precision runtime extension. It does not rewrite ComfyUI source files, mutate weight.data or retain a full FP32 weight mirror. Preserve the FP32 safety paths during server validation.

Thanks to ComfyUI, MiniMax and Amduraznak/minimax-h3-fp16-fix for the Custom Node pattern and related mixed-precision design. This is not an official release of those projects. ComfyUI-derived code and this extension are distributed under GPL-3.0-only; see LICENSE. The acknowledged project retains its own license.
