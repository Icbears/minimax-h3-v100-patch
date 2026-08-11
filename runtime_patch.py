"""Runtime MiniMax H3 V100 mixed-precision patch for ComfyUI.

The profile deliberately starts from ComfyUI's FP32 H3 path. Only the fused
QKV projection and text/video attention of the main DiT blocks are moved to
FP16. Q/K RMSNorm, RoPE, target/reference-audio attention, attention output
projection, MLP, AdaLN, the residual stream, token refiners, and final heads
remain on their source/FP32 paths.

No ComfyUI source file is modified. The patch is installed when ComfyUI imports
this custom-node package and is removed by deleting the package and restarting.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import contextvars
import inspect
from typing import Any, Iterable

import torch

import comfy.ldm.minimax.model as mm
import comfy.model_management as model_management
import comfy.quant_ops as quant_ops
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention


PROFILE_ID = "minimax-h3-v100-fp32-base-fp16-hotspots-audio-safe-v1"
PROFILE_LABEL = "FP32 base + FP16 QKV/text-video attention + FP32 audio attention"
PACKAGE_VERSION = "0.1.2"
_MODULE_PATCH_MARKER = "_minimax_h3_v100_custom_node_profile"
_ATTENTION_ENABLE_FLAG = "_minimax_h3_v100_fp16_hotspots"
_QKV_WEIGHT_FLAG = "_minimax_h3_v100_resident_fp16_qkv"
_LAYOUT_RANGES_ATTR = "_minimax_h3_v100_fp32_audio_ranges"
_OPTIONS_RANGES_KEY = "minimax_h3_fp32_audio_ranges"

_AUDIO_RANGES: contextvars.ContextVar[tuple[tuple[int, int], ...]] = contextvars.ContextVar(
    "minimax_h3_v100_audio_ranges", default=()
)
_CAPTURE_LAYOUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "minimax_h3_v100_capture_layout", default=False
)

_ORIGINAL_ATTENTION_FORWARD = None
_ORIGINAL_BLOCK_FORWARD = None
_ORIGINAL_MLP_FORWARD = None
_ORIGINAL_MODEL_INIT = None
_ORIGINAL_MODEL_FORWARD = None
_ORIGINAL_LAYOUT_INIT = None
_WARNED: set[str] = set()


def _log(message: str) -> None:
    print(f"[MiniMax H3 V100] {message}")


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _log(f"WARNING: {message}")


def _signature_has(function: Any, required: Iterable[str]) -> bool:
    try:
        names = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return all(name in names for name in required)


def _source_contains(function: Any, required: Iterable[str]) -> bool:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return all(token in source for token in required)


def _validate_runtime_shape() -> tuple[bool, str]:
    """Refuse unknown or already-rewritten internals before changing classes."""

    required_types = ("Attention", "DiTBlock", "MLP", "MiniMaxH3Model", "PackedLayout")
    missing = [name for name in required_types if not hasattr(mm, name)]
    if missing:
        return False, "unsupported H3 module; missing " + ", ".join(missing)

    attention_forward = mm.Attention.forward
    block_forward = mm.DiTBlock.forward
    mlp_forward = mm.MLP.forward
    model_init = mm.MiniMaxH3Model.__init__
    model_forward = getattr(mm.MiniMaxH3Model, "_forward", None)
    layout_init = mm.PackedLayout.__init__

    if model_forward is None:
        return False, "unsupported H3 module; MiniMaxH3Model._forward is absent"
    if not _signature_has(attention_forward, ("self", "x", "rope_freqs", "transformer_options")):
        return False, "unsupported Attention.forward signature"
    if not _signature_has(model_init, ("self",)):
        return False, "unsupported MiniMaxH3Model.__init__ signature"
    if not _signature_has(model_forward, ("self", "x", "context", "transformer_options", "minimax_payload")):
        return False, "unsupported MiniMaxH3Model._forward signature"
    if not _signature_has(layout_init, ("self", "text_len", "latent_t", "audio_t")):
        return False, "unsupported PackedLayout.__init__ signature"

    # Replacing a function already owned by another extension would make load
    # order decide numerical behavior. Disable instead of silently stacking.
    if getattr(attention_forward, "__module__", None) != mm.__name__:
        return False, "Attention.forward is already owned by another runtime extension"
    if getattr(block_forward, "__module__", None) != mm.__name__:
        return False, "DiTBlock.forward is already owned by another runtime extension"
    if getattr(mlp_forward, "__module__", None) != mm.__name__:
        return False, "MLP.forward is already owned by another runtime extension"
    if getattr(model_init, "__module__", None) != mm.__name__:
        return False, "MiniMaxH3Model.__init__ is already owned by another runtime extension"
    if getattr(model_forward, "__module__", None) != mm.__name__:
        return False, "MiniMaxH3Model._forward is already owned by another runtime extension"

    if not _source_contains(
        attention_forward,
        ("qkv_proj", "q_norm", "k_norm", "optimized_attention", "out_proj"),
    ):
        return False, "unsupported Attention.forward implementation"
    if _source_contains(attention_forward, ("fp16_qkv",)):
        return False, "a source-level MiniMax H3 V100 patch is already present"
    if not _source_contains(model_forward, ("PackedLayout", "layout.segments", "minimax_payload")):
        return False, "unsupported MiniMaxH3Model._forward implementation"

    return True, "supported current MiniMax H3 structure"


def _ranges_from_layout(layout: Any) -> tuple[tuple[int, int], ...]:
    if layout is None:
        return ()
    cached = getattr(layout, _LAYOUT_RANGES_ATTR, None)
    if cached is not None:
        return tuple(cached)
    segments = getattr(layout, "segments", ())
    ranges = []
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            continue
        start, stop, kind = segment
        if kind in ("audio", "ref_audio"):
            ranges.append((int(start), int(stop)))
    return tuple(ranges)


def _normalize_ranges(ranges: Any, sequence_length: int) -> tuple[tuple[int, int], ...]:
    normalized = []
    try:
        for start, stop in ranges:
            start = int(start)
            stop = int(stop)
            if not 0 <= start < stop <= sequence_length:
                return ()
            normalized.append((start, stop))
    except (TypeError, ValueError):
        return ()
    return tuple(normalized)


def _patched_layout_init(self, *args, **kwargs):
    _ORIGINAL_LAYOUT_INIT(self, *args, **kwargs)
    ranges = _ranges_from_layout(self)
    setattr(self, _LAYOUT_RANGES_ATTR, ranges)
    if _CAPTURE_LAYOUT.get():
        _AUDIO_RANGES.set(ranges)


def _patched_model_forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    **kwargs,
):
    payload = minimax_payload or {}
    layout = payload.get("layout") if hasattr(payload, "get") else None
    ranges_token = _AUDIO_RANGES.set(_ranges_from_layout(layout))
    capture_token = _CAPTURE_LAYOUT.set(layout is None)
    try:
        return _ORIGINAL_MODEL_FORWARD(
            self,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    finally:
        _CAPTURE_LAYOUT.reset(capture_token)
        _AUDIO_RANGES.reset(ranges_token)


def _attention_shape_supported(attention: Any) -> bool:
    required = ("heads", "head_dim", "qkv_proj", "q_norm", "k_norm", "out_proj")
    return all(hasattr(attention, name) for name in required)


def _patched_model_init(self, *args, **kwargs):
    _ORIGINAL_MODEL_INIT(self, *args, **kwargs)
    if (
        mm.Attention.forward is not _patched_attention_forward
        or mm.DiTBlock.forward is not _ORIGINAL_BLOCK_FORWARD
        or mm.MLP.forward is not _ORIGINAL_MLP_FORWARD
    ):
        _warn_once(
            "late-runtime-conflict",
            "another H3 runtime extension loaded after this one; the FP16-hotspot profile stays disabled",
        )
        return
    blocks = tuple(getattr(self, "blocks", ()))
    if not blocks:
        _warn_once("missing-blocks", "H3 main DiT blocks were not found; this model instance stays unmodified")
        return
    if not all(hasattr(block, "attn") and _attention_shape_supported(block.attn) for block in blocks):
        _warn_once(
            "unsupported-block",
            "the main DiT attention structure is unfamiliar; this model instance stays unmodified",
        )
        return

    for block in blocks:
        setattr(block.attn, _ATTENTION_ENABLE_FLAG, True)
        setattr(block.attn, _QKV_WEIGHT_FLAG, True)
    _log(f"enabled {PROFILE_LABEL} on {len(blocks)} main DiT blocks")


def _prepare_qkv_weight(attention: Any) -> bool:
    """Convert each main-block QKV weight once; fail closed for other dtypes."""

    if not getattr(attention, _QKV_WEIGHT_FLAG, False):
        return True
    try:
        weight = attention.qkv_proj.weight
        if weight.dtype == torch.float32:
            with torch.no_grad():
                weight.data = weight.data.to(dtype=torch.float16)
        elif weight.dtype != torch.float16:
            _warn_once(
                "qkv-weight-dtype",
                "a main QKV weight is neither FP32 nor FP16; the runtime profile is disabled for that block",
            )
            setattr(attention, _ATTENTION_ENABLE_FLAG, False)
            return False
    except (AttributeError, RuntimeError, TypeError) as exc:
        _warn_once(
            "qkv-weight-conversion",
            f"resident FP16 QKV conversion failed ({exc}); the runtime profile is disabled for that block",
        )
        setattr(attention, _ATTENTION_ENABLE_FLAG, False)
        return False
    return True


def _patched_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    enabled = (
        getattr(self, _ATTENTION_ENABLE_FLAG, False)
        and getattr(getattr(x, "device", None), "type", None) == "cuda"
        and x.dtype == torch.float32
        and not model_management.in_training
    )
    if not enabled:
        return _ORIGINAL_ATTENTION_FORWARD(
            self,
            x,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
    if not _prepare_qkv_weight(self):
        return _ORIGINAL_ATTENTION_FORWARD(
            self,
            x,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )

    sequence_length = x.shape[0]
    residual_dtype = x.dtype
    options = transformer_options if hasattr(transformer_options, "get") else {}
    raw_ranges = options.get(_OPTIONS_RANGES_KEY, ()) or _AUDIO_RANGES.get()
    audio_ranges = _normalize_ranges(raw_ranges, sequence_length)

    # QKV uses FP16 Tensor Cores. Q/K immediately return to FP32 for the
    # stability-sensitive RMSNorm and RoPE operations; V stays FP16 until an
    # FP32 attention path promotes it.
    q, k, v = self.qkv_proj(x.to(dtype=torch.float16)).split(
        self.heads * self.head_dim, dim=-1
    )
    q = q.to(dtype=torch.float32)
    k = k.to(dtype=torch.float32)
    v = v.view(sequence_length, self.heads, self.head_dim)

    if rope_freqs is not None:
        q = q.view(1, sequence_length, self.heads, self.head_dim)
        k = k.view(1, sequence_length, self.heads, self.head_dim)
        qw = model_management.cast_to(self.q_norm.weight, dtype=q.dtype, device=x.device)
        kw = model_management.cast_to(self.k_norm.weight, dtype=k.dtype, device=x.device)
        rope = rope_freqs.to(dtype=q.dtype) if rope_freqs.dtype != q.dtype else rope_freqs
        rotation_dim = rope.shape[-3] * 2
        quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope,
            qw,
            kw,
            epsilon=self.q_norm.eps,
            rot_dim=rotation_dim,
        )
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(sequence_length, self.heads, self.head_dim))
        k = self.k_norm(k.view(sequence_length, self.heads, self.head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    if not audio_ranges:
        # H3 is an audio-video model. Missing/invalid layout metadata must not
        # silently send audio through the FP16 attention path. We keep only the
        # already-completed FP16 QKV optimization and fail closed to full FP32
        # attention for this call.
        _warn_once(
            "missing-audio-ranges",
            "audio row metadata was unavailable; falling back to full FP32 attention for safety",
        )
        out = optimized_attention(
            q,
            k,
            v.to(dtype=torch.float32),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=options,
        )
    else:
        # Compute the packed sequence once in FP16, then recompute only target
        # and reference-audio query rows with FP32 attention and overwrite them.
        out = optimized_attention(
            q.to(dtype=torch.float16),
            k.to(dtype=torch.float16),
            v.to(dtype=torch.float16),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=options,
        ).to(dtype=residual_dtype)
        audio_v = v.to(dtype=torch.float32)
        for start, stop in audio_ranges:
            out[:, start:stop] = attention_pytorch(
                q[:, :, start:stop],
                k,
                audio_v,
                self.heads,
                mask=None,
                skip_reshape=True,
            )

    # The output projection receives FP32 and therefore stays on the existing
    # source/FP32 path. MLP, AdaLN and residual code are never monkey-patched.
    return self.out_proj(out.squeeze(0))


def install_patch() -> dict[str, Any]:
    """Install the profile once, or return a safe disabled status."""

    existing = getattr(mm, _MODULE_PATCH_MARKER, None)
    if existing:
        if existing == PROFILE_ID:
            return {
                "installed": True,
                "profile": PROFILE_ID,
                "version": PACKAGE_VERSION,
                "reason": "already installed",
            }
        reason = f"another MiniMax H3 runtime profile is already installed: {existing}"
        _log(f"disabled: {reason}")
        return {
            "installed": False,
            "profile": PROFILE_ID,
            "version": PACKAGE_VERSION,
            "reason": reason,
        }

    supported, reason = _validate_runtime_shape()
    if not supported:
        _log(f"disabled: {reason}; no ComfyUI source file was changed")
        return {
            "installed": False,
            "profile": PROFILE_ID,
            "version": PACKAGE_VERSION,
            "reason": reason,
        }

    global _ORIGINAL_ATTENTION_FORWARD
    global _ORIGINAL_BLOCK_FORWARD
    global _ORIGINAL_MLP_FORWARD
    global _ORIGINAL_MODEL_INIT
    global _ORIGINAL_MODEL_FORWARD
    global _ORIGINAL_LAYOUT_INIT

    _ORIGINAL_ATTENTION_FORWARD = mm.Attention.forward
    _ORIGINAL_BLOCK_FORWARD = mm.DiTBlock.forward
    _ORIGINAL_MLP_FORWARD = mm.MLP.forward
    _ORIGINAL_MODEL_INIT = mm.MiniMaxH3Model.__init__
    _ORIGINAL_MODEL_FORWARD = mm.MiniMaxH3Model._forward
    _ORIGINAL_LAYOUT_INIT = mm.PackedLayout.__init__

    _patched_attention_forward._minimax_h3_v100_profile = PROFILE_ID
    _patched_model_init._minimax_h3_v100_profile = PROFILE_ID
    _patched_model_forward._minimax_h3_v100_profile = PROFILE_ID
    _patched_layout_init._minimax_h3_v100_profile = PROFILE_ID

    try:
        mm.Attention.forward = _patched_attention_forward
        mm.MiniMaxH3Model.__init__ = _patched_model_init
        mm.MiniMaxH3Model._forward = _patched_model_forward
        mm.PackedLayout.__init__ = _patched_layout_init
        setattr(mm, _MODULE_PATCH_MARKER, PROFILE_ID)
    except Exception:
        mm.Attention.forward = _ORIGINAL_ATTENTION_FORWARD
        mm.MiniMaxH3Model.__init__ = _ORIGINAL_MODEL_INIT
        mm.MiniMaxH3Model._forward = _ORIGINAL_MODEL_FORWARD
        mm.PackedLayout.__init__ = _ORIGINAL_LAYOUT_INIT
        if hasattr(mm, _MODULE_PATCH_MARKER):
            delattr(mm, _MODULE_PATCH_MARKER)
        raise

    _log(f"v{PACKAGE_VERSION} runtime profile installed: {PROFILE_LABEL}")
    _log("no --fp16-unet flag is required; source files remain untouched")
    return {
        "installed": True,
        "profile": PROFILE_ID,
        "version": PACKAGE_VERSION,
        "reason": reason,
    }


PATCH_STATUS = install_patch()
