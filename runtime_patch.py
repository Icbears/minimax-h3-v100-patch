"""MiniMax H3 V100 v0.1.3 mixed-precision profile for ComfyUI.

This release registers FP16 as a supported inference dtype for MiniMax H3
before the model is instantiated. Main-block weights are therefore created,
loaded, prefetched, and offloaded through ComfyUI's native FP16 path. The main
DiT residual stream is promoted to FP32 while its attention/MLP branches run in
FP16. Target/reference-audio attention and SwiGLU intermediates run in FP32;
scaled output projections return FP32 values to the residual stream.

No ComfyUI source file is modified and no --fp16-unet flag is required.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import contextvars
import inspect
from typing import Any, Iterable

import torch
from torch.nn import functional as F

import comfy.ldm.minimax.model as mm
import comfy.model_management as model_management
import comfy.quant_ops as quant_ops
import comfy.supported_models as supported_models
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention


PROFILE_ID = "minimax-h3-v100-v013-native-fp16-branches"
PROFILE_LABEL = "v0.1.3: native FP16 storage/branches + FP32 safety islands"
PACKAGE_VERSION = "0.1.3"
OUT_PROJ_SCALE = 64.0
MLP_FC2_SCALE = 256.0

_MODULE_PATCH_MARKER = "_minimax_h3_v100_custom_node_profile"
_SUPPORTED_MODEL_PATCH_MARKER = "_minimax_h3_v100_supported_dtype_profile"
_BLOCK_ENABLE_FLAG = "_minimax_h3_v100_v013_fp32_residual"
_ATTENTION_ENABLE_FLAG = "_minimax_h3_v100_v013_audio_safe_attention"
_MLP_ENABLE_FLAG = "_minimax_h3_v100_v013_scaled_mlp"
_FINAL_ENABLE_FLAG = "_minimax_h3_v100_v013_fp32_final"
_CONDITION_WRAPPED_FLAG = "_minimax_h3_v100_v013_fp32_condition"
_LAYOUT_RANGES_ATTR = "_minimax_h3_v100_fp32_audio_ranges"
_OPTIONS_RANGES_KEY = "minimax_h3_fp32_audio_ranges"

_AUDIO_RANGES: contextvars.ContextVar[tuple[tuple[int, int], ...]] = contextvars.ContextVar(
    "minimax_h3_v100_v013_audio_ranges", default=()
)
_CAPTURE_LAYOUT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "minimax_h3_v100_v013_capture_layout", default=False
)

_ORIGINAL_SUPPORTED_DTYPES = None
_ORIGINAL_ATTENTION_FORWARD = None
_ORIGINAL_BLOCK_FORWARD = None
_ORIGINAL_MLP_FORWARD = None
_ORIGINAL_FINAL_FORWARD = None
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


def _target_device_supported() -> tuple[bool, str]:
    """Limit automatic native-FP16 selection to the Volta/V100 target."""

    try:
        if not torch.cuda.is_available():
            return False, "CUDA is unavailable"
        capability = tuple(torch.cuda.get_device_capability())
    except (AssertionError, RuntimeError, TypeError) as exc:
        return False, f"CUDA capability detection failed: {exc}"
    if capability != (7, 0):
        return False, f"CUDA capability {capability[0]}.{capability[1]} is not the V100 target 7.0"
    return True, "CUDA capability 7.0 (Volta/V100)"


def _validate_runtime_shape() -> tuple[bool, str]:
    """Refuse unknown source structures or pre-existing runtime rewrites."""

    required_types = (
        "Attention",
        "DiTBlock",
        "FinalLayer",
        "MLP",
        "MiniMaxH3Model",
        "PackedLayout",
    )
    missing = [name for name in required_types if not hasattr(mm, name)]
    if missing:
        return False, "unsupported H3 module; missing " + ", ".join(missing)
    if not hasattr(supported_models, "MiniMaxH3"):
        return False, "unsupported ComfyUI; supported_models.MiniMaxH3 is absent"
    if not all(hasattr(mm, name) for name in ("_mod_scale_shift", "_mod_gate")):
        return False, "unsupported H3 module; residual modulation helpers are absent"

    attention_forward = mm.Attention.forward
    block_forward = mm.DiTBlock.forward
    final_forward = mm.FinalLayer.forward
    mlp_forward = mm.MLP.forward
    model_init = mm.MiniMaxH3Model.__init__
    model_forward = getattr(mm.MiniMaxH3Model, "_forward", None)
    layout_init = mm.PackedLayout.__init__

    if model_forward is None:
        return False, "unsupported H3 module; MiniMaxH3Model._forward is absent"
    if not _signature_has(attention_forward, ("self", "x", "rope_freqs", "transformer_options")):
        return False, "unsupported Attention.forward signature"
    if not _signature_has(block_forward, ("self", "x", "t_emb", "mod_segments", "rope_freqs")):
        return False, "unsupported DiTBlock.forward signature"
    if not _signature_has(mlp_forward, ("self", "x")):
        return False, "unsupported MLP.forward signature"
    if not _signature_has(final_forward, ("self", "x", "t_emb", "video_seg", "audio_seg")):
        return False, "unsupported FinalLayer.forward signature"
    if not _signature_has(model_init, ("self", "dtype", "operations")):
        return False, "unsupported MiniMaxH3Model.__init__ signature"
    if not _signature_has(model_forward, ("self", "x", "context", "transformer_options", "minimax_payload")):
        return False, "unsupported MiniMaxH3Model._forward signature"
    if not _signature_has(layout_init, ("self", "text_len", "latent_t", "audio_t")):
        return False, "unsupported PackedLayout.__init__ signature"

    owned_methods = (
        ("Attention.forward", attention_forward),
        ("DiTBlock.forward", block_forward),
        ("FinalLayer.forward", final_forward),
        ("MLP.forward", mlp_forward),
        ("MiniMaxH3Model.__init__", model_init),
        ("MiniMaxH3Model._forward", model_forward),
        ("PackedLayout.__init__", layout_init),
    )
    for label, function in owned_methods:
        if getattr(function, "__module__", None) != mm.__name__:
            return False, f"{label} is already owned by another runtime extension"

    if not _source_contains(
        attention_forward,
        ("qkv_proj", "q_norm", "k_norm", "optimized_attention", "out_proj"),
    ):
        return False, "unsupported Attention.forward implementation"
    if _source_contains(attention_forward, ("fp16_qkv",)):
        return False, "a source-level MiniMax H3 V100 patch is already present"
    if not _source_contains(
        block_forward,
        ("adaln_proj", "norm1", "norm2", "_mod_scale_shift", "_mod_gate"),
    ):
        return False, "unsupported DiTBlock.forward implementation"
    if not _source_contains(mlp_forward, ("fc1", "fc2", "swiglu")):
        return False, "unsupported MLP.forward implementation"
    if not _source_contains(final_forward, ("adaln_proj", "video_out", "audio_out")):
        return False, "unsupported FinalLayer.forward implementation"
    if not _source_contains(model_forward, ("PackedLayout", "layout.segments", "minimax_payload")):
        return False, "unsupported MiniMaxH3Model._forward implementation"

    dtype_owner = supported_models.MiniMaxH3
    if getattr(dtype_owner, "__module__", None) != supported_models.__name__:
        return False, "supported_models.MiniMaxH3 is already replaced by another extension"
    current_dtypes = tuple(getattr(dtype_owner, "supported_inference_dtypes", ()))
    if torch.float16 in current_dtypes:
        return False, "MiniMaxH3 already exposes FP16 inference through another source or extension"
    if torch.bfloat16 not in current_dtypes or torch.float32 not in current_dtypes:
        return False, "unsupported MiniMaxH3 inference-dtype declaration"

    return True, "supported ComfyUI 0.33.2 MiniMax H3 structure"


def _ranges_from_layout(layout: Any) -> tuple[tuple[int, int], ...]:
    if layout is None:
        return ()
    cached = getattr(layout, _LAYOUT_RANGES_ATTR, None)
    if cached is not None:
        return tuple(cached)
    ranges = []
    for segment in getattr(layout, "segments", ()):
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
        result = _ORIGINAL_MODEL_FORWARD(
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
    return result


def _attention_shape_supported(attention: Any) -> bool:
    required = ("heads", "head_dim", "qkv_proj", "q_norm", "k_norm", "out_proj")
    return all(hasattr(attention, name) for name in required)


def _block_shape_supported(block: Any) -> bool:
    return (
        all(hasattr(block, name) for name in ("attn", "mlp", "adaln_proj", "norm1", "norm2"))
        and _attention_shape_supported(block.attn)
        and all(hasattr(block.mlp, name) for name in ("fc1", "fc2"))
    )


def _final_shape_supported(final_layer: Any) -> bool:
    return all(
        hasattr(final_layer, name)
        for name in ("norm", "adaln_proj", "video_out", "audio_out")
    )


def _wrap_condition_projection(model: Any) -> bool:
    projection = getattr(model, "condition_proj", None)
    if projection is None or not hasattr(projection, "forward"):
        return False
    if getattr(projection, _CONDITION_WRAPPED_FLAG, False):
        return True
    original_forward = projection.forward

    def fp32_condition_forward(value, _forward=original_forward):
        return _forward(value.to(dtype=torch.float32))

    fp32_condition_forward._minimax_h3_v100_profile = PROFILE_ID
    projection.forward = fp32_condition_forward
    setattr(projection, _CONDITION_WRAPPED_FLAG, True)
    return True


def _patched_model_init(self, *args, **kwargs):
    _ORIGINAL_MODEL_INIT(self, *args, **kwargs)
    if (
        mm.Attention.forward is not _patched_attention_forward
        or mm.DiTBlock.forward is not _patched_block_forward
        or mm.MLP.forward is not _patched_mlp_forward
        or mm.FinalLayer.forward is not _patched_final_forward
    ):
        _warn_once(
            "late-runtime-conflict",
            "another H3 runtime extension loaded after v0.1.3; new H3 instances stay unmodified",
        )
        return
    if getattr(self, "dtype", None) != torch.float16:
        _warn_once(
            "model-not-native-fp16",
            f"H3 was instantiated as {getattr(self, 'dtype', None)}, not FP16; v0.1.3 is inactive for this instance",
        )
        return

    blocks = tuple(getattr(self, "blocks", ()))
    if not blocks or not all(_block_shape_supported(block) for block in blocks):
        _warn_once(
            "unsupported-block",
            "the main H3 block structure is unfamiliar; v0.1.3 is inactive for this instance",
        )
        return
    final_layer = getattr(self, "final_layer", None)
    if final_layer is None or not _final_shape_supported(final_layer):
        _warn_once(
            "unsupported-final-layer",
            "the H3 FinalLayer structure is unfamiliar; v0.1.3 is inactive for this instance",
        )
        return
    if not _wrap_condition_projection(self):
        _warn_once(
            "missing-condition-proj",
            "condition_proj could not be wrapped for FP32 overflow safety; v0.1.3 is inactive for this instance",
        )
        return

    for block in blocks:
        setattr(block, _BLOCK_ENABLE_FLAG, True)
        setattr(block.attn, _ATTENTION_ENABLE_FLAG, True)
        setattr(block.mlp, _MLP_ENABLE_FLAG, True)
    setattr(final_layer, _FINAL_ENABLE_FLAG, True)
    _log(f"enabled {PROFILE_LABEL} on {len(blocks)} main DiT blocks")


def _patched_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    enabled = (
        getattr(self, _ATTENTION_ENABLE_FLAG, False)
        and getattr(getattr(x, "device", None), "type", None) == "cuda"
        and x.dtype == torch.float16
        and not model_management.in_training
    )
    if not enabled:
        return _ORIGINAL_ATTENTION_FORWARD(
            self,
            x,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )

    sequence_length = x.shape[0]
    options = transformer_options if hasattr(transformer_options, "get") else {}
    raw_ranges = options.get(_OPTIONS_RANGES_KEY, ()) or _AUDIO_RANGES.get()
    audio_ranges = _normalize_ranges(raw_ranges, sequence_length)

    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
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
        _warn_once(
            "missing-audio-ranges",
            "audio row metadata was unavailable; this call falls back to full FP32 attention",
        )
        out = optimized_attention(
            q.to(dtype=torch.float32),
            k.to(dtype=torch.float32),
            v.to(dtype=torch.float32),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=options,
        )
    else:
        out = optimized_attention(
            q,
            k,
            v,
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=options,
        )
        audio_q = q.to(dtype=torch.float32)
        audio_k = k.to(dtype=torch.float32)
        audio_v = v.to(dtype=torch.float32)
        for start, stop in audio_ranges:
            audio_out = attention_pytorch(
                audio_q[:, :, start:stop],
                audio_k,
                audio_v,
                self.heads,
                mask=None,
                skip_reshape=True,
            )
            out[:, start:stop] = audio_out.to(dtype=out.dtype)

    projected = self.out_proj(
        (out.squeeze(0) / OUT_PROJ_SCALE).to(dtype=torch.float16)
    )
    return projected.to(dtype=torch.float32).mul_(OUT_PROJ_SCALE)


def _patched_mlp_forward(self, x):
    enabled = (
        getattr(self, _MLP_ENABLE_FLAG, False)
        and getattr(getattr(x, "device", None), "type", None) == "cuda"
        and x.dtype == torch.float16
        and not model_management.in_training
    )
    if not enabled:
        return _ORIGINAL_MLP_FORWARD(self, x)

    gate, value = self.fc1(x).chunk(2, dim=-1)
    hidden = F.silu(gate.to(dtype=torch.float32)).mul_(value.to(dtype=torch.float32))
    projected = self.fc2(
        (hidden / MLP_FC2_SCALE).to(dtype=torch.float16)
    )
    return projected.to(dtype=torch.float32).mul_(MLP_FC2_SCALE)


def _patched_final_forward(self, x, t_emb, video_seg, audio_seg):
    enabled = (
        getattr(self, _FINAL_ENABLE_FLAG, False)
        and getattr(getattr(x, "device", None), "type", None) == "cuda"
        and not model_management.in_training
    )
    if not enabled:
        return _ORIGINAL_FINAL_FORWARD(self, x, t_emb, video_seg, audio_seg)

    x = x.to(dtype=torch.float32)
    t_emb = t_emb.to(dtype=torch.float32)
    shift, scale = self.adaln_proj(t_emb)
    va, vb, vrow = video_seg
    aa, ab, arow = audio_seg
    hv = self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]
    ha = self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]
    return self.video_out(hv.to(dtype=torch.float32)), self.audio_out(ha.to(dtype=torch.float32))


def _patched_block_forward(
    self,
    x,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options={},
):
    enabled = (
        getattr(self, _BLOCK_ENABLE_FLAG, False)
        and getattr(getattr(x, "device", None), "type", None) == "cuda"
        and not model_management.in_training
    )
    if not enabled:
        return _ORIGINAL_BLOCK_FORWARD(
            self,
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )

    if x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    t_emb = t_emb.to(dtype=torch.float32)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = mm._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments).to(
        dtype=torch.float16
    )
    attention_out = self.attn(
        h,
        rope_freqs=rope_freqs,
        transformer_options=transformer_options,
    )
    x = mm._mod_gate(
        x,
        gate_msa,
        attention_out.to(dtype=torch.float32),
        mod_segments,
    )
    h = mm._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments).to(
        dtype=torch.float16
    )
    mlp_out = self.mlp(h)
    return mm._mod_gate(
        x,
        gate_mlp,
        mlp_out.to(dtype=torch.float32),
        mod_segments,
    )


def install_patch() -> dict[str, Any]:
    """Install native-FP16 selection and FP32 safety islands once."""

    existing = getattr(mm, _MODULE_PATCH_MARKER, None)
    dtype_existing = getattr(supported_models.MiniMaxH3, _SUPPORTED_MODEL_PATCH_MARKER, None)
    if existing or dtype_existing:
        if existing == PROFILE_ID and dtype_existing == PROFILE_ID:
            return {
                "installed": True,
                "profile": PROFILE_ID,
                "version": PACKAGE_VERSION,
                "reason": "already installed",
            }
        reason = f"another MiniMax H3 runtime profile is installed: {existing or dtype_existing}"
        _log(f"disabled: {reason}")
        return {
            "installed": False,
            "profile": PROFILE_ID,
            "version": PACKAGE_VERSION,
            "reason": reason,
        }

    target_ok, target_reason = _target_device_supported()
    if not target_ok:
        _log(f"disabled: {target_reason}; no dtype declaration or source file was changed")
        return {
            "installed": False,
            "profile": PROFILE_ID,
            "version": PACKAGE_VERSION,
            "reason": target_reason,
        }

    supported, reason = _validate_runtime_shape()
    if not supported:
        _log(f"disabled: {reason}; no dtype declaration or source file was changed")
        return {
            "installed": False,
            "profile": PROFILE_ID,
            "version": PACKAGE_VERSION,
            "reason": reason,
        }

    global _ORIGINAL_SUPPORTED_DTYPES
    global _ORIGINAL_ATTENTION_FORWARD
    global _ORIGINAL_BLOCK_FORWARD
    global _ORIGINAL_MLP_FORWARD
    global _ORIGINAL_FINAL_FORWARD
    global _ORIGINAL_MODEL_INIT
    global _ORIGINAL_MODEL_FORWARD
    global _ORIGINAL_LAYOUT_INIT

    _ORIGINAL_SUPPORTED_DTYPES = tuple(supported_models.MiniMaxH3.supported_inference_dtypes)
    _ORIGINAL_ATTENTION_FORWARD = mm.Attention.forward
    _ORIGINAL_BLOCK_FORWARD = mm.DiTBlock.forward
    _ORIGINAL_MLP_FORWARD = mm.MLP.forward
    _ORIGINAL_FINAL_FORWARD = mm.FinalLayer.forward
    _ORIGINAL_MODEL_INIT = mm.MiniMaxH3Model.__init__
    _ORIGINAL_MODEL_FORWARD = mm.MiniMaxH3Model._forward
    _ORIGINAL_LAYOUT_INIT = mm.PackedLayout.__init__

    for function in (
        _patched_attention_forward,
        _patched_block_forward,
        _patched_mlp_forward,
        _patched_final_forward,
        _patched_model_init,
        _patched_model_forward,
        _patched_layout_init,
    ):
        function._minimax_h3_v100_profile = PROFILE_ID

    try:
        supported_models.MiniMaxH3.supported_inference_dtypes = [
            torch.float16,
            *_ORIGINAL_SUPPORTED_DTYPES,
        ]
        setattr(supported_models.MiniMaxH3, _SUPPORTED_MODEL_PATCH_MARKER, PROFILE_ID)
        mm.Attention.forward = _patched_attention_forward
        mm.DiTBlock.forward = _patched_block_forward
        mm.MLP.forward = _patched_mlp_forward
        mm.FinalLayer.forward = _patched_final_forward
        mm.MiniMaxH3Model.__init__ = _patched_model_init
        mm.MiniMaxH3Model._forward = _patched_model_forward
        mm.PackedLayout.__init__ = _patched_layout_init
        setattr(mm, _MODULE_PATCH_MARKER, PROFILE_ID)
    except Exception:
        supported_models.MiniMaxH3.supported_inference_dtypes = list(_ORIGINAL_SUPPORTED_DTYPES)
        if hasattr(supported_models.MiniMaxH3, _SUPPORTED_MODEL_PATCH_MARKER):
            delattr(supported_models.MiniMaxH3, _SUPPORTED_MODEL_PATCH_MARKER)
        mm.Attention.forward = _ORIGINAL_ATTENTION_FORWARD
        mm.DiTBlock.forward = _ORIGINAL_BLOCK_FORWARD
        mm.MLP.forward = _ORIGINAL_MLP_FORWARD
        mm.FinalLayer.forward = _ORIGINAL_FINAL_FORWARD
        mm.MiniMaxH3Model.__init__ = _ORIGINAL_MODEL_INIT
        mm.MiniMaxH3Model._forward = _ORIGINAL_MODEL_FORWARD
        mm.PackedLayout.__init__ = _ORIGINAL_LAYOUT_INIT
        if hasattr(mm, _MODULE_PATCH_MARKER):
            delattr(mm, _MODULE_PATCH_MARKER)
        raise

    _log(f"v{PACKAGE_VERSION} runtime profile installed: {PROFILE_LABEL}")
    _log(f"registered native FP16 H3 loading for {target_reason}; no --fp16-unet flag is required")
    return {
        "installed": True,
        "profile": PROFILE_ID,
        "version": PACKAGE_VERSION,
        "reason": reason,
    }


PATCH_STATUS = install_patch()
