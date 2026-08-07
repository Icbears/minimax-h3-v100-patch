#!/usr/bin/env python3
"""Shared, dependency-free installer for MiniMax H3 V100 mixed precision.

SPDX-License-Identifier: GPL-3.0-only

This installer modifies ComfyUI's ``comfy/ldm/minimax/model.py``. The
replacement Attention method is based on the upstream ComfyUI implementation
and was modified on 2026-08-05 to add the tested V100 FP16 compute islands.
"""

from __future__ import print_function

import argparse
import ast
import codecs
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


TARGET = Path("comfy/ldm/minimax/model.py")

KNOWN_NORMALIZED_SHA256 = {
    "origin": {
        "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024",
    },
    "te": {
        "57414496df9c4bef974acbb486d9a6cf77f1ad087bcd10364a8994b67b5be949",
    },
}

BACKUP_SUFFIX = {
    "origin": ".v100_origin.bak",
    "te": ".v100_te.bak",
}

RUN_BLOCKS_ANCHOR = (
    "    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, "
    "transformer_options, start=0, end=None):"
)
BLOCK_LOOP_ANCHOR = '("block_loop", 0) in blocks_replace'
STOCK_LOOP_ANCHOR = (
    "        prefetch_queue = comfy.model_prefetch.make_prefetch_queue("
    "list(self.blocks), device, transformer_options)"
)
PATCH_METHOD_MARKER = 'getattr(self, "fp16_qkv", False)'
PATCH_FLAG_MARKER = "            block.attn.fp16_qkv = True"
FINAL_LAYER_MARKER = "        self.final_layer = FinalLayer("

TE_FORWARD_ANCHOR = '''    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
'''

TE_RUN_BLOCKS_METHOD = '''    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        end = len(self.blocks) if end is None else end
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), h.device, transformer_options)
        for i in range(start, end):
            block = self.blocks[i]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
        return h

'''

ORIGIN_BLOCK_LOOP = '''        # blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
'''

TE_BLOCK_LOOP = '''        # blocks
        # blocks (TE-Speed-MiniMaxH3-OSS hook)
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
        if ("block_loop", 0) in blocks_replace:
            def block_loop_wrap(args):
                return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                args["transformer_options"], args.get("start", 0), args.get("end"))}
            h = blocks_replace[("block_loop", 0)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options, "cache_ranges": cache_ranges, "block_count": len(self.blocks)},
                {"original_block": block_loop_wrap})["img"]
        else:
            h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
'''


ORIGINAL_ATTENTION_METHOD = '''    def forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))
'''


PATCHED_ATTENTION_METHOD = '''    def forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        residual_dtype = x.dtype
        use_fp16 = (
            getattr(self, "fp16_qkv", False)
            and x.device.type == "cuda"
            and x.dtype == torch.float32
        )

        # MiniMax H3 V100 acceleration (tested Plan 2): QKV projection and
        # attention use FP16 Tensor Cores. Q/K return to FP32 immediately for
        # RMSNorm and RoPE. Output projection, MLP, AdaLN and residual stay FP32.
        proj_x = x.half() if use_fp16 else x
        q, k, v = self.qkv_proj(proj_x).split(self.heads * self.head_dim, dim=-1)
        if use_fp16:
            q = q.float()
            k = k.float()
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, dtype=q.dtype, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, dtype=k.dtype, device=x.device)
            rope = rope_freqs.to(q.dtype) if rope_freqs.dtype != q.dtype else rope_freqs
            rot = rope.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if use_fp16:
            out = optimized_attention(
                q.half(), k.half(), v.half(), self.heads, mask=None,
                skip_reshape=True, transformer_options=transformer_options,
            ).to(residual_dtype)
        else:
            out = optimized_attention(
                q, k, v, self.heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options,
            )
        return self.out_proj(out.squeeze(0))
'''


ENABLE_BLOCKS = '''        # MiniMax H3 V100 Plan 2: enable FP16 only on the 50 main DiT attentions.
        # The two token-refiner blocks intentionally retain the source compute dtype.
        for block in self.blocks:
            block.attn.fp16_qkv = True
'''


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalized_sha256(text):
    return sha256_bytes(text.encode("utf-8"))


def read_model(path):
    raw = path.read_bytes()
    has_bom = raw.startswith(codecs.BOM_UTF8)
    decoded = raw.decode("utf-8-sig")
    crlf = raw.count(b"\r\n")
    bare_lf = raw.count(b"\n") - crlf
    if crlf and bare_lf:
        raise SystemExit("error: mixed CRLF/LF newlines are not patched automatically")
    newline = "\r\n" if crlf else "\n"
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    return raw, normalized, newline, has_bom


def encode_model(text, newline, has_bom):
    raw = text.replace("\n", newline).encode("utf-8")
    return (codecs.BOM_UTF8 + raw) if has_bom else raw


def detect_variant(text):
    has_run_blocks = RUN_BLOCKS_ANCHOR in text
    has_block_loop = BLOCK_LOOP_ANCHOR in text
    if has_run_blocks != has_block_loop:
        return "partial_te"
    if has_run_blocks:
        return "te"
    if STOCK_LOOP_ANCHOR in text:
        return "origin"
    return "unknown"


def patch_state(text):
    has_method = PATCH_METHOD_MARKER in text
    has_flag = PATCH_FLAG_MARKER in text
    if has_method and has_flag:
        return "patched"
    if has_method or has_flag:
        return "partial"
    return "unpatched"


def validate_target(text, expected_variant, allow_patched=True):
    variant = detect_variant(text)
    state = patch_state(text)
    if variant == "partial_te":
        raise SystemExit("error: partial TE hook installation detected; restore a clean model.py first")
    if variant != expected_variant:
        if variant in ("origin", "te"):
            other = "patch_te_v100" if variant == "te" else "patch_h3_origin_v100"
            raise SystemExit(
                "error: wrong model variant: found {0}, expected {1}. Use {2}."
                .format(variant, expected_variant, other)
            )
        raise SystemExit("error: unsupported MiniMax H3 model.py layout; no file was changed")
    if state == "partial":
        raise SystemExit("error: partial V100 patch detected; restore a clean backup first")
    if state == "patched" and not allow_patched:
        raise SystemExit("error: backup unexpectedly already contains the V100 patch")
    return state


def apply_patch(text, expected_variant):
    validate_target(text, expected_variant, allow_patched=False)
    if text.count(ORIGINAL_ATTENTION_METHOD) != 1:
        raise SystemExit(
            "error: the upstream Attention.forward body does not match the supported version; "
            "no file was changed"
        )
    if text.count(FINAL_LAYER_MARKER) != 1:
        raise SystemExit("error: expected exactly one FinalLayer construction anchor")

    patched = text.replace(ORIGINAL_ATTENTION_METHOD, PATCHED_ATTENTION_METHOD, 1)
    patched = patched.replace(FINAL_LAYER_MARKER, ENABLE_BLOCKS + FINAL_LAYER_MARKER, 1)
    ast.parse(patched)
    if validate_target(patched, expected_variant) != "patched":
        raise SystemExit("error: internal patch validation failed")
    return patched


def install_te_hooks(text):
    """Promote the supported clean origin layout to the TE-hooked layout."""
    validate_target(text, "origin", allow_patched=False)
    if text.count(TE_FORWARD_ANCHOR) != 1:
        raise SystemExit("error: expected exactly one MiniMaxH3Model.forward anchor")
    if text.count(ORIGIN_BLOCK_LOOP) != 1:
        raise SystemExit(
            "error: the origin block loop does not match the supported TE conversion; "
            "no file was changed"
        )

    promoted = text.replace(TE_FORWARD_ANCHOR, TE_RUN_BLOCKS_METHOD + TE_FORWARD_ANCHOR, 1)
    promoted = promoted.replace(ORIGIN_BLOCK_LOOP, TE_BLOCK_LOOP, 1)
    ast.parse(promoted)
    if detect_variant(promoted) != "te" or patch_state(promoted) != "unpatched":
        raise SystemExit("error: internal origin-to-TE conversion validation failed")
    return promoted


def candidate_from_base(base):
    base = Path(base).expanduser()
    candidates = []
    if base.is_file():
        candidates.append(base)
    else:
        candidates.extend([
            base / TARGET,
            base / "ComfyUI" / TARGET,
            base / "ComfyUI_windows_portable" / "ComfyUI" / TARGET,
        ])
    return candidates


def auto_roots():
    roots = []
    env_root = os.environ.get("COMFYUI_PATH")
    if env_root:
        roots.append(Path(env_root))

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        current = start
        for _ in range(6):
            roots.append(current)
            if current.parent == current:
                break
            current = current.parent

    if os.name == "nt":
        for drive in "CDEFGH":
            root = Path(drive + ":/")
            roots.extend([
                root / "ComfyUI",
                root / "ComfyUI_windows_portable",
                root / "ComfyUI-aki-v3",
            ])
    return roots


def unique_existing(candidates):
    seen = set()
    result = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            key = os.path.normcase(str(candidate))
            if key not in seen and candidate.is_file():
                seen.add(key)
                result.append(candidate)
        except (OSError, RuntimeError):
            continue
    return result


def find_model_file(expected_variant, explicit=None, allow_origin_to_te=False):
    if explicit:
        candidates = unique_existing(candidate_from_base(explicit))
        if not candidates:
            raise SystemExit("error: no comfy/ldm/minimax/model.py found under {0}".format(explicit))
    else:
        raw_candidates = []
        for root in auto_roots():
            raw_candidates.extend(candidate_from_base(root))
        candidates = unique_existing(raw_candidates)

    accepted_variants = {expected_variant}
    if expected_variant == "te" and allow_origin_to_te:
        accepted_variants.add("origin")

    matches = []
    wrong = []
    for candidate in candidates:
        try:
            _, text, _, _ = read_model(candidate)
            variant = detect_variant(text)
            if variant in accepted_variants:
                matches.append(candidate)
            else:
                wrong.append((candidate, variant))
        except (OSError, UnicodeError, SystemExit):
            continue

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listing = "\n  ".join(str(p) for p in matches)
        raise SystemExit(
            "error: multiple matching ComfyUI installs found; pass a path explicitly:\n  " + listing
        )
    if wrong:
        listing = "\n  ".join("{0} ({1})".format(p, v) for p, v in wrong)
        raise SystemExit(
            "error: found model.py files, but none is a supported {0} target:\n  {1}"
            .format(expected_variant, listing)
        )
    raise SystemExit(
        "error: could not locate ComfyUI. Drag model.py onto the BAT file or pass "
        "--comfy-ui <ComfyUI root>."
    )


def atomic_replace(path, data):
    mode = stat.S_IMODE(path.stat().st_mode)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=path.name + ".", suffix=".tmp", dir=str(path.parent), delete=False
        ) as handle:
            temp_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, str(path))
        temp_name = None
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def print_status(target, variant, state, raw, text):
    known = normalized_sha256(text) in KNOWN_NORMALIZED_SHA256[variant]
    if state == "patched":
        known_label = "n/a (active file is already patched)"
    else:
        known_label = "yes" if known else "no (structural anchors still checked)"
    print("Target:             {0}".format(target))
    print("Model variant:      {0}".format(variant))
    print("V100 patch state:   {0}".format(state))
    print("Input SHA-256:      {0}".format(sha256_bytes(raw)))
    print("Known source build: {0}".format(known_label))


def run(expected_variant, argv=None, allow_origin_to_te=False):
    title = "TE-hooked H3" if expected_variant == "te" else "official H3 origin"
    parser = argparse.ArgumentParser(
        description="Patch {0} for the tested MiniMax H3 V100 Plan 2 precision split".format(title)
    )
    parser.add_argument(
        "path", nargs="?", help="ComfyUI root or model.py; enables drag-and-drop onto the BAT launcher"
    )
    parser.add_argument("--comfy-ui", help="explicit ComfyUI root directory")
    parser.add_argument("--model-file", help="explicit comfy/ldm/minimax/model.py path")
    parser.add_argument("--check", action="store_true", help="report status without changing files")
    parser.add_argument("--revert", action="store_true", help="restore this patcher's exact backup")
    args = parser.parse_args(argv)

    explicit_values = [x for x in (args.path, args.comfy_ui, args.model_file) if x]
    if len(explicit_values) > 1:
        raise SystemExit("error: use only one of path, --comfy-ui, or --model-file")
    explicit = explicit_values[0] if explicit_values else None
    target = find_model_file(expected_variant, explicit, allow_origin_to_te=allow_origin_to_te)
    backup = target.with_name(target.name + BACKUP_SUFFIX[expected_variant])

    raw, text, newline, has_bom = read_model(target)
    active_variant = detect_variant(text)
    promoting_origin = (
        expected_variant == "te" and allow_origin_to_te and active_variant == "origin"
    )
    validation_variant = "origin" if promoting_origin else expected_variant
    state = validate_target(text, validation_variant)
    print_status(target, active_variant, state, raw, text)

    if promoting_origin:
        print("TE hook state:      origin detected; automatic TE promotion is available")

    if args.check:
        return 0

    if args.revert:
        if promoting_origin:
            if state == "unpatched":
                print("Already restored to the clean origin file; no file was changed.")
                return 0
            raise SystemExit(
                "error: the active origin file is not a TE patch; use restore_h3_origin_v100"
            )
        if state != "patched":
            raise SystemExit("error: refusing to revert because the active file is not V100-patched")
        if not backup.is_file():
            raise SystemExit("error: no backup at {0}".format(backup))
        backup_raw, backup_text, _, _ = read_model(backup)
        backup_variant = detect_variant(backup_text)
        if backup_variant not in ("origin", "te"):
            raise SystemExit("error: backup is not a supported origin or TE model.py")
        validate_target(backup_text, backup_variant, allow_patched=False)
        ast.parse(backup_text)
        atomic_replace(target, backup_raw)
        restored_raw, restored_text, _, _ = read_model(target)
        validate_target(restored_text, backup_variant, allow_patched=False)
        print("Restored SHA-256:   {0}".format(sha256_bytes(restored_raw)))
        print("Restored variant:   {0}".format(backup_variant))
        print("Restored from:      {0}".format(backup))
        print("Restart ComfyUI before the next inference run.")
        return 0

    if state == "patched":
        if promoting_origin:
            raise SystemExit(
                "error: origin V100 patch detected; restore it before installing the TE version"
            )
        print("Already patched; no file was changed.")
        return 0

    if backup.is_file():
        backup_raw, backup_text, _, _ = read_model(backup)
        backup_variant = detect_variant(backup_text)
        if backup_variant != active_variant:
            raise SystemExit(
                "error: existing backup is for {0}, but the active clean model is {1}. "
                "Move the stale backup out of the way before patching: {2}"
                .format(backup_variant, active_variant, backup)
            )
        validate_target(backup_text, backup_variant, allow_patched=False)
        if backup_raw != raw:
            raise SystemExit(
                "error: existing backup differs from the active clean model. Move the stale backup "
                "out of the way before patching: {0}".format(backup)
            )
        print("Backup already matches the active source: {0}".format(backup))
    else:
        shutil.copy2(str(target), str(backup))
        print("Backup written:     {0}".format(backup))

    patch_source = install_te_hooks(text) if promoting_origin else text
    if promoting_origin:
        print("TE hooks installed:  origin -> te (in the verified output transaction)")
    patched_text = apply_patch(patch_source, expected_variant)
    patched_raw = encode_model(patched_text, newline, has_bom)
    atomic_replace(target, patched_raw)

    verify_raw, verify_text, _, _ = read_model(target)
    ast.parse(verify_text)
    if validate_target(verify_text, expected_variant) != "patched":
        raise SystemExit("error: post-write verification failed; restore the backup")
    if verify_raw != patched_raw:
        raise SystemExit("error: post-write byte verification failed; restore the backup")

    print("Patched SHA-256:    {0}".format(sha256_bytes(verify_raw)))
    print("Patch profile:      Plan 2 (QKV + attention FP16; Q/K norm/RoPE + residual FP32)")
    print("Restart ComfyUI before benchmarking.")
    return 0


def main(expected_variant, allow_origin_to_te=False):
    try:
        return run(expected_variant, allow_origin_to_te=allow_origin_to_te)
    except KeyboardInterrupt:
        print("error: interrupted; no further action taken", file=sys.stderr)
        return 130
