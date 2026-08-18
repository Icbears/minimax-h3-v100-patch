"""Dependency-free structural and dtype-flow tests for v0.1.3."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest


class _FakeWeight:
    def __init__(self, dtype="float16"):
        self.dtype = dtype


class _TraceTensor:
    def __init__(self, name, dtype, trace=None, shape=(8, 16)):
        self.name = name
        self.dtype = dtype
        self.trace = [] if trace is None else trace
        self.device = types.SimpleNamespace(type="cuda")
        self.shape = shape

    def to(self, dtype=None, **kwargs):
        self.trace.append((f"{self.name}.to", dtype))
        return _TraceTensor(self.name, dtype, self.trace, self.shape)

    def chunk(self, count, dim=-1):
        self.trace.append((f"{self.name}.chunk", count, dim))
        return (
            _TraceTensor("gate", self.dtype, self.trace, self.shape),
            _TraceTensor("value", self.dtype, self.trace, self.shape),
        )

    def __truediv__(self, scale):
        self.trace.append((f"{self.name}/", scale, self.dtype))
        return _TraceTensor(self.name, self.dtype, self.trace, self.shape)

    def __getitem__(self, key):
        self.trace.append((f"{self.name}[]", key, self.dtype))
        return _TraceTensor(self.name, self.dtype, self.trace, self.shape)

    def __add__(self, other):
        other_dtype = other.dtype if isinstance(other, _TraceTensor) else None
        self.trace.append((f"{self.name}+", other_dtype, self.dtype))
        return _TraceTensor(self.name, self.dtype, self.trace, self.shape)

    def __radd__(self, other):
        self.trace.append((f"+{self.name}", other, self.dtype))
        return _TraceTensor(self.name, self.dtype, self.trace, self.shape)

    def __mul__(self, other):
        other_dtype = other.dtype if isinstance(other, _TraceTensor) else None
        self.trace.append((f"{self.name}*", other_dtype, self.dtype))
        return _TraceTensor(self.name, self.dtype, self.trace, self.shape)

    def mul_(self, other):
        if isinstance(other, _TraceTensor):
            self.trace.append((f"{self.name}.mul_", other.name, self.dtype, other.dtype))
        else:
            self.trace.append((f"{self.name}.mul_", other, self.dtype))
        return self


class _FakeLinear:
    def __init__(self, name="linear", dtype="float16"):
        self.name = name
        self.weight = _FakeWeight(dtype)

    def __call__(self, value):
        value.trace.append((self.name, value.dtype))
        return _TraceTensor(f"{self.name}_out", value.dtype, value.trace, value.shape)


class _FakeNorm:
    def __init__(self):
        self.weight = _FakeWeight()
        self.eps = 1e-5

    def __call__(self, value):
        value.trace.append(("norm", value.dtype))
        return value


class _FakeProjection:
    def __init__(self):
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value.dtype)
        return value

    def __call__(self, value):
        return self.forward(value)


class _FakeAttention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 4
        self.qkv_proj = _FakeLinear("qkv")
        self.q_norm = _FakeNorm()
        self.k_norm = _FakeNorm()
        self.out_proj = _FakeLinear("out_proj")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        qkv_proj = self.qkv_proj
        q_norm = self.q_norm
        k_norm = self.k_norm
        optimized_attention = transformer_options
        out_proj = self.out_proj
        return x, rope_freqs, qkv_proj, q_norm, k_norm, optimized_attention, out_proj


class _FakeMLP:
    def __init__(self):
        self.fc1 = _FakeLinear("fc1")
        self.fc2 = _FakeLinear("fc2")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        fc1 = self.fc1
        fc2 = self.fc2
        swiglu = "swiglu"
        return x, fc1, fc2, swiglu


class _FakeAdaln:
    def __call__(self, value):
        return tuple(_TraceTensor(f"adaln_{index}", value.dtype, value.trace) for index in range(6))


class _FakeFinalAdaln:
    def __call__(self, value):
        return (
            _TraceTensor("final_shift", value.dtype, value.trace),
            _TraceTensor("final_scale", value.dtype, value.trace),
        )


class _FakeFinalLayer:
    def __init__(self):
        self.norm = _FakeNorm()
        self.adaln_proj = _FakeFinalAdaln()
        self.video_out = _FakeLinear("video_out")
        self.audio_out = _FakeLinear("audio_out")

    def forward(self, x, t_emb, video_seg, audio_seg):
        adaln_proj = self.adaln_proj
        video_out = self.video_out
        audio_out = self.audio_out
        return x, t_emb, video_seg, audio_seg, adaln_proj, video_out, audio_out


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttention()
        self.mlp = _FakeMLP()
        self.adaln_proj = _FakeAdaln()
        self.norm1 = _FakeNorm()
        self.norm2 = _FakeNorm()


class _FakeDiTBlock(_FakeBlock):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        adaln_proj = self.adaln_proj
        norm1 = self.norm1
        norm2 = self.norm2
        _mod_scale_shift = mod_segments
        _mod_gate = transformer_options
        return x, t_emb, rope_freqs, adaln_proj, norm1, norm2, _mod_scale_shift, _mod_gate


class _FakeTokenRefiner:
    def __init__(self):
        self.blocks = [_FakeBlock(), _FakeBlock()]


class _FakePackedLayout:
    def __init__(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=None,
        refs=None,
        frame_count=None,
    ):
        self.segments = [
            (0, text_len, "text"),
            (text_len, text_len + audio_t, "ref_audio"),
            (text_len + audio_t, text_len + audio_t * 3, "audio"),
            (text_len + audio_t * 3, text_len + audio_t * 3 + latent_t, "video"),
        ]


class _FakeModel:
    def __init__(self, num_layers=3, dtype=None, operations=None, **kwargs):
        self.dtype = dtype
        self.operations = operations
        self.condition_proj = _FakeProjection()
        self.blocks = [_FakeBlock() for _ in range(num_layers)]
        self.token_refiner = _FakeTokenRefiner()
        self.final_layer = _FakeFinalLayer()

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        payload = minimax_payload or {}
        layout = payload.get("layout")
        PackedLayout = _FakePackedLayout
        if layout is None:
            layout = PackedLayout(2, 3, 4, 5, 6)
        probe = transformer_options.get("probe")
        if probe is not None:
            return probe()
        return PackedLayout, layout.segments, minimax_payload, x, timestep, context


class _FakeSupportedMiniMaxH3:
    supported_inference_dtypes = ["bfloat16", "float32"]


class RuntimePatchTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = dict(sys.modules)
        self.runtime_path = Path(__file__).resolve().parents[1] / "runtime_patch.py"
        self.fake_methods = {
            "attention_forward": _FakeAttention.forward,
            "block_forward": _FakeDiTBlock.forward,
            "final_forward": _FakeFinalLayer.forward,
            "mlp_forward": _FakeMLP.forward,
            "layout_init": _FakePackedLayout.__init__,
            "model_init": _FakeModel.__init__,
            "model_forward": _FakeModel._forward,
        }

    def tearDown(self):
        _FakeAttention.forward = self.fake_methods["attention_forward"]
        _FakeDiTBlock.forward = self.fake_methods["block_forward"]
        _FakeFinalLayer.forward = self.fake_methods["final_forward"]
        _FakeMLP.forward = self.fake_methods["mlp_forward"]
        _FakePackedLayout.__init__ = self.fake_methods["layout_init"]
        _FakeModel.__init__ = self.fake_methods["model_init"]
        _FakeModel._forward = self.fake_methods["model_forward"]
        _FakeSupportedMiniMaxH3.supported_inference_dtypes = ["bfloat16", "float32"]
        for name in (
            "_minimax_h3_v100_supported_dtype_profile",
            "_minimax_h3_v100_custom_node_profile",
        ):
            if hasattr(_FakeSupportedMiniMaxH3, name):
                delattr(_FakeSupportedMiniMaxH3, name)
        sys.modules.clear()
        sys.modules.update(self.saved_modules)

    def _install_fake_modules(self, capability=(7, 0), cuda_available=True, fp16_predeclared=False):
        torch = types.ModuleType("torch")
        torch.__path__ = []
        torch.float16 = "float16"
        torch.bfloat16 = "bfloat16"
        torch.float32 = "float32"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_capability=lambda: capability,
        )
        torch_nn = types.ModuleType("torch.nn")
        torch_nn.__path__ = []
        torch_functional = types.ModuleType("torch.nn.functional")
        torch_functional.silu = lambda value: value
        torch_nn.functional = torch_functional
        torch.nn = torch_nn

        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        ldm = types.ModuleType("comfy.ldm")
        ldm.__path__ = []
        minimax = types.ModuleType("comfy.ldm.minimax")
        minimax.__path__ = []
        modules = types.ModuleType("comfy.ldm.modules")
        modules.__path__ = []
        attention = types.ModuleType("comfy.ldm.modules.attention")
        attention.attention_pytorch = lambda *args, **kwargs: None
        attention.optimized_attention = lambda *args, **kwargs: None
        management = types.ModuleType("comfy.model_management")
        management.in_training = False
        management.cast_to = lambda value, **kwargs: value
        quant_ops = types.ModuleType("comfy.quant_ops")
        quant_ops.ck = types.SimpleNamespace(rms_rope_split_half_=lambda *args, **kwargs: None)

        model = types.ModuleType("comfy.ldm.minimax.model")
        for cls in (
            _FakeAttention,
            _FakeDiTBlock,
            _FakeFinalLayer,
            _FakeMLP,
            _FakePackedLayout,
            _FakeModel,
        ):
            cls.__module__ = model.__name__
        for function in self.fake_methods.values():
            function.__module__ = model.__name__
        model.Attention = _FakeAttention
        model.DiTBlock = _FakeDiTBlock
        model.FinalLayer = _FakeFinalLayer
        model.MLP = _FakeMLP
        model.PackedLayout = _FakePackedLayout
        model.MiniMaxH3Model = _FakeModel

        def mod_scale_shift(value, shift, scale, segments):
            value.trace.append(("mod_scale_shift", value.dtype))
            return value

        def mod_gate(value, gate, other, segments):
            value.trace.append(("mod_gate", value.dtype, other.dtype))
            return value

        model._mod_scale_shift = mod_scale_shift
        model._mod_gate = mod_gate

        supported = types.ModuleType("comfy.supported_models")
        _FakeSupportedMiniMaxH3.__module__ = supported.__name__
        _FakeSupportedMiniMaxH3.supported_inference_dtypes = (
            ["float16", "bfloat16", "float32"]
            if fp16_predeclared
            else ["bfloat16", "float32"]
        )
        supported.MiniMaxH3 = _FakeSupportedMiniMaxH3

        sys.modules.update(
            {
                "torch": torch,
                "torch.nn": torch_nn,
                "torch.nn.functional": torch_functional,
                "comfy": comfy,
                "comfy.ldm": ldm,
                "comfy.ldm.minimax": minimax,
                "comfy.ldm.minimax.model": model,
                "comfy.ldm.modules": modules,
                "comfy.ldm.modules.attention": attention,
                "comfy.model_management": management,
                "comfy.quant_ops": quant_ops,
                "comfy.supported_models": supported,
            }
        )
        comfy.ldm = ldm
        comfy.model_management = management
        comfy.quant_ops = quant_ops
        comfy.supported_models = supported
        ldm.minimax = minimax
        minimax.model = model
        return model, supported

    def _load_runtime(self):
        spec = importlib.util.spec_from_file_location("test_minimax_h3_v013_runtime", self.runtime_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_registers_native_fp16_and_marks_only_native_fp16_main_blocks(self):
        model_module, supported = self._install_fake_modules()
        runtime = self._load_runtime()
        self.assertTrue(runtime.PATCH_STATUS["installed"])
        self.assertEqual(runtime.PACKAGE_VERSION, "0.1.3")
        self.assertEqual(
            supported.MiniMaxH3.supported_inference_dtypes,
            ["float16", "bfloat16", "float32"],
        )

        model = model_module.MiniMaxH3Model(num_layers=3, dtype="float16", operations=object())
        for block in model.blocks:
            self.assertTrue(getattr(block, runtime._BLOCK_ENABLE_FLAG))
            self.assertTrue(getattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))
            self.assertTrue(getattr(block.mlp, runtime._MLP_ENABLE_FLAG))
        self.assertTrue(getattr(model.final_layer, runtime._FINAL_ENABLE_FLAG))
        for block in model.token_refiner.blocks:
            self.assertFalse(hasattr(block, runtime._BLOCK_ENABLE_FLAG))
            self.assertFalse(hasattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))
            self.assertFalse(hasattr(block.mlp, runtime._MLP_ENABLE_FLAG))

        value = _TraceTensor("condition", "float16")
        result = model.condition_proj(value)
        self.assertEqual(model.condition_proj.inputs, ["float32"])
        self.assertEqual(result.dtype, "float32")

    def test_fp32_forced_model_instance_stays_unmodified(self):
        model_module, _ = self._install_fake_modules()
        runtime = self._load_runtime()
        model = model_module.MiniMaxH3Model(dtype="float32", operations=object())
        for block in model.blocks:
            self.assertFalse(hasattr(block, runtime._BLOCK_ENABLE_FLAG))
            self.assertFalse(hasattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))
            self.assertFalse(hasattr(block.mlp, runtime._MLP_ENABLE_FLAG))
        self.assertFalse(hasattr(model.final_layer, runtime._FINAL_ENABLE_FLAG))
        self.assertFalse(hasattr(model.condition_proj, runtime._CONDITION_WRAPPED_FLAG))

    def test_audio_ranges_are_captured_and_reset(self):
        model_module, _ = self._install_fake_modules()
        runtime = self._load_runtime()
        model = model_module.MiniMaxH3Model(dtype="float16", operations=object())
        layout = model_module.PackedLayout(2, 3, 4, 5, 6)
        expected = ((2, 8), (8, 20))
        self.assertEqual(runtime._ranges_from_layout(layout), expected)
        inside = model._forward(
            None,
            None,
            None,
            transformer_options={"probe": runtime._AUDIO_RANGES.get},
            minimax_payload={"layout": layout},
        )
        self.assertEqual(inside, expected)
        self.assertEqual(runtime._AUDIO_RANGES.get(), ())

    def test_block_trace_promotes_residual_and_demotes_only_branches(self):
        _, _ = self._install_fake_modules()
        runtime = self._load_runtime()
        trace = []

        class Branch:
            def __init__(self, name):
                self.name = name

            def __call__(self, value, **kwargs):
                trace.append((self.name, value.dtype))
                return _TraceTensor(f"{self.name}_out", "float32", trace)

        block = types.SimpleNamespace(
            adaln_proj=_FakeAdaln(),
            norm1=_FakeNorm(),
            norm2=_FakeNorm(),
            attn=Branch("attention"),
            mlp=Branch("mlp"),
        )
        setattr(block, runtime._BLOCK_ENABLE_FLAG, True)
        x = _TraceTensor("residual", "float16", trace)
        t_emb = _TraceTensor("t_emb", "float16", trace)
        result = runtime._patched_block_forward(block, x, t_emb, (), None, {})
        self.assertEqual(result.dtype, "float32")
        self.assertIn(("residual.to", "float32"), trace)
        self.assertIn(("t_emb.to", "float32"), trace)
        self.assertIn(("attention", "float16"), trace)
        self.assertIn(("mlp", "float16"), trace)
        self.assertIn(("mod_gate", "float32", "float32"), trace)

    def test_mlp_trace_uses_native_fp16_gemms_and_fp32_swiglu(self):
        _, _ = self._install_fake_modules()
        runtime = self._load_runtime()
        mlp = _FakeMLP()
        setattr(mlp, runtime._MLP_ENABLE_FLAG, True)
        trace = []

        def traced_silu(value):
            trace.append(("silu", value.dtype))
            return _TraceTensor("silu_gate", value.dtype, trace)

        runtime.F.silu = traced_silu
        result = runtime._patched_mlp_forward(mlp, _TraceTensor("x", "float16", trace))
        self.assertEqual(result.dtype, "float32")
        self.assertEqual(
            trace,
            [
                ("fc1", "float16"),
                ("fc1_out.chunk", 2, -1),
                ("gate.to", "float32"),
                ("silu", "float32"),
                ("value.to", "float32"),
                ("silu_gate.mul_", "value", "float32", "float32"),
                ("silu_gate/", 256.0, "float32"),
                ("silu_gate.to", "float16"),
                ("fc2", "float16"),
                ("fc2_out.to", "float32"),
                ("fc2_out.mul_", 256.0, "float32"),
            ],
        )

    def test_final_layer_promotes_modulation_and_both_heads_to_fp32(self):
        _, _ = self._install_fake_modules()
        runtime = self._load_runtime()
        final_layer = _FakeFinalLayer()
        setattr(final_layer, runtime._FINAL_ENABLE_FLAG, True)
        trace = []
        video, audio = runtime._patched_final_forward(
            final_layer,
            _TraceTensor("final_x", "float16", trace),
            _TraceTensor("final_t", "float16", trace),
            (0, 4, 0),
            (4, 8, 1),
        )
        self.assertEqual(video.dtype, "float32")
        self.assertEqual(audio.dtype, "float32")
        self.assertIn(("final_x.to", "float32"), trace)
        self.assertIn(("final_t.to", "float32"), trace)
        self.assertIn(("video_out", "float32"), trace)
        self.assertIn(("audio_out", "float32"), trace)

    def test_non_v100_device_disables_before_dtype_registration(self):
        _, supported = self._install_fake_modules(capability=(8, 0))
        runtime = self._load_runtime()
        self.assertFalse(runtime.PATCH_STATUS["installed"])
        self.assertIn("not the V100 target", runtime.PATCH_STATUS["reason"])
        self.assertEqual(supported.MiniMaxH3.supported_inference_dtypes, ["bfloat16", "float32"])

    def test_predeclared_fp16_is_treated_as_conflict(self):
        _, supported = self._install_fake_modules(fp16_predeclared=True)
        runtime = self._load_runtime()
        self.assertFalse(runtime.PATCH_STATUS["installed"])
        self.assertIn("already exposes FP16", runtime.PATCH_STATUS["reason"])
        self.assertEqual(
            supported.MiniMaxH3.supported_inference_dtypes,
            ["float16", "bfloat16", "float32"],
        )

    def test_install_is_idempotent(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        second = runtime.install_patch()
        self.assertTrue(second["installed"])
        self.assertEqual(second["version"], "0.1.3")
        self.assertEqual(second["reason"], "already installed")

    def test_late_conflict_keeps_new_instances_unmodified(self):
        model_module, _ = self._install_fake_modules()
        runtime = self._load_runtime()

        def foreign_mlp(self, x):
            return x

        foreign_mlp.__module__ = "another_extension"
        model_module.MLP.forward = foreign_mlp
        model = model_module.MiniMaxH3Model(dtype="float16", operations=object())
        for block in model.blocks:
            self.assertFalse(hasattr(block, runtime._BLOCK_ENABLE_FLAG))

    def test_source_contract_has_no_lazy_weight_mutation(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        source = self.runtime_path.read_text(encoding="utf-8")
        self.assertEqual(runtime.OUT_PROJ_SCALE, 64.0)
        self.assertEqual(runtime.MLP_FC2_SCALE, 256.0)
        self.assertNotIn("weight.data", source)
        self.assertNotIn("_prepare_fp16_weights", source)
        self.assertIn("supported_inference_dtypes", source)
        self.assertIn("audio_q = q.to(dtype=torch.float32)", source)
        self.assertIn("out.squeeze(0) / OUT_PROJ_SCALE", source)
        self.assertIn("hidden / MLP_FC2_SCALE", source)
        self.assertIn("t_emb = t_emb.to(dtype=torch.float32)", source)
        self.assertIn("self.video_out(hv.to(dtype=torch.float32))", source)
        self.assertIn("self.audio_out(ha.to(dtype=torch.float32))", source)

    def test_runtime_code_has_no_source_file_write_api(self):
        source = self.runtime_path.read_text(encoding="utf-8")
        for forbidden in ("open(", "write_text(", "write_bytes(", "shutil.", "os.rename"):
            self.assertNotIn(forbidden, source)

    def test_runtime_contains_no_allocator_debug_telemetry(self):
        source = self.runtime_path.read_text(encoding="utf-8")
        self.assertNotIn("empty_cache", source)
        self.assertNotIn("reset_peak_memory_stats", source)
        self.assertNotIn("_memory_report", source)
        self.assertNotIn("memory_allocated", source)
        self.assertNotIn("memory_reserved", source)
        self.assertNotIn("max_memory_allocated", source)
        self.assertNotIn("mem_get_info", source)
        self.assertNotIn("oom-forward", source)

    def test_core_compute_functions_match_promoted_l3_release(self):
        candidate = ast.parse(self.runtime_path.read_text(encoding="utf-8"))
        expected = {
            "_patched_attention_forward": "d423ee3d5e20c19d4219c133a7c32459ef37a3d3135df6436ba47a4d90d4adb3",
            "_patched_mlp_forward": "aa8b8c08b1d41d7dd0a0ef3bfd124f72293130697865014443016ad8b336f9ef",
            "_patched_final_forward": "96582fee3f8b3c661aabbaeb484f2e8e4efc2dfd589801e3648e413cd590d394",
            "_patched_block_forward": "866c73f012a7f6c62e1c84a207832c184ff1d335f499d72da3200845b79496ee",
        }

        for name, expected_hash in expected.items():
            node = next(
                item
                for item in candidate.body
                if isinstance(item, ast.FunctionDef) and item.name == name
            )
            payload = ast.dump(node, include_attributes=False).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main()
