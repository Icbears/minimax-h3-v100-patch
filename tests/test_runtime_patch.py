"""Dependency-free structural tests for the custom-node runtime installer."""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest


class _FakeWeightData:
    def to(self, dtype=None):
        return self


class _FakeWeight:
    dtype = "float32"
    data = _FakeWeightData()


class _FakeLinear:
    weight = _FakeWeight()


class _FakeNorm:
    weight = _FakeWeight()
    eps = 1e-5


class _FakeAttention:
    def __init__(self):
        self.heads = 2
        self.head_dim = 4
        self.qkv_proj = _FakeLinear()
        self.q_norm = _FakeNorm()
        self.k_norm = _FakeNorm()
        self.out_proj = _FakeLinear()

    def forward(self, x, rope_freqs=None, transformer_options={}):
        # Tokens below intentionally model the supported upstream structure.
        qkv_proj = self.qkv_proj
        q_norm = self.q_norm
        k_norm = self.k_norm
        optimized_attention = transformer_options
        out_proj = self.out_proj
        return x, rope_freqs, qkv_proj, q_norm, k_norm, optimized_attention, out_proj


class _FakeBlock:
    def __init__(self):
        self.attn = _FakeAttention()


class _FakeTokenRefiner:
    def __init__(self):
        self.blocks = [_FakeBlock(), _FakeBlock()]


class _FakeDiTBlock(_FakeBlock):
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        return x


class _FakeMLP:
    def forward(self, x):
        return x


class _FakePackedLayout:
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        self.segments = [
            (0, text_len, "text"),
            (text_len, text_len + audio_t * 2, "audio"),
            (text_len + audio_t * 2, text_len + audio_t * 2 + latent_t, "video"),
        ]


class _FakeModel:
    def __init__(self, num_layers=3, **kwargs):
        self.blocks = [_FakeBlock() for _ in range(num_layers)]
        self.token_refiner = _FakeTokenRefiner()

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        payload = minimax_payload or {}
        layout = payload.get("layout")
        PackedLayout = _FakePackedLayout
        if layout is None:
            layout = PackedLayout(2, 3, 4, 5, 6)
        probe = transformer_options.get("probe")
        if probe is not None:
            return probe()
        return PackedLayout, layout.segments, minimax_payload, x, timestep, context, transformer_options


class RuntimePatchTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = dict(sys.modules)
        self.runtime_path = Path(__file__).resolve().parents[1] / "runtime_patch.py"
        self.fake_methods = {
            "attention_forward": _FakeAttention.forward,
            "block_forward": _FakeDiTBlock.forward,
            "mlp_forward": _FakeMLP.forward,
            "layout_init": _FakePackedLayout.__init__,
            "model_init": _FakeModel.__init__,
            "model_forward": _FakeModel._forward,
        }

    def tearDown(self):
        _FakeAttention.forward = self.fake_methods["attention_forward"]
        _FakeDiTBlock.forward = self.fake_methods["block_forward"]
        _FakeMLP.forward = self.fake_methods["mlp_forward"]
        _FakePackedLayout.__init__ = self.fake_methods["layout_init"]
        _FakeModel.__init__ = self.fake_methods["model_init"]
        _FakeModel._forward = self.fake_methods["model_forward"]
        sys.modules.clear()
        sys.modules.update(self.saved_modules)

    def _install_fake_modules(self):
        torch = types.ModuleType("torch")
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.no_grad = contextlib.nullcontext

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
        for cls in (_FakeAttention, _FakeDiTBlock, _FakeMLP, _FakePackedLayout, _FakeModel):
            cls.__module__ = model.__name__
        _FakeAttention.forward.__module__ = model.__name__
        _FakeDiTBlock.forward.__module__ = model.__name__
        _FakeMLP.forward.__module__ = model.__name__
        _FakePackedLayout.__init__.__module__ = model.__name__
        _FakeModel.__init__.__module__ = model.__name__
        _FakeModel._forward.__module__ = model.__name__
        model.Attention = _FakeAttention
        model.DiTBlock = _FakeDiTBlock
        model.MLP = _FakeMLP
        model.PackedLayout = _FakePackedLayout
        model.MiniMaxH3Model = _FakeModel

        sys.modules.update(
            {
                "torch": torch,
                "comfy": comfy,
                "comfy.ldm": ldm,
                "comfy.ldm.minimax": minimax,
                "comfy.ldm.minimax.model": model,
                "comfy.ldm.modules": modules,
                "comfy.ldm.modules.attention": attention,
                "comfy.model_management": management,
                "comfy.quant_ops": quant_ops,
            }
        )
        comfy.ldm = ldm
        comfy.model_management = management
        comfy.quant_ops = quant_ops
        ldm.minimax = minimax
        minimax.model = model
        return model

    def _load_runtime(self):
        spec = importlib.util.spec_from_file_location("test_minimax_h3_runtime_patch", self.runtime_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_install_marks_only_main_blocks_and_captures_audio_ranges(self):
        model_module = self._install_fake_modules()
        runtime = self._load_runtime()

        self.assertTrue(runtime.PATCH_STATUS["installed"])
        self.assertEqual(runtime.PATCH_STATUS["version"], "0.1.2")
        model = model_module.MiniMaxH3Model(num_layers=3)
        self.assertEqual(len(model.blocks), 3)
        for block in model.blocks:
            self.assertTrue(getattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))
            self.assertTrue(getattr(block.attn, runtime._QKV_WEIGHT_FLAG))
        for block in model.token_refiner.blocks:
            self.assertFalse(hasattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))

        layout = model_module.PackedLayout(2, 3, 4, 5, 6)
        self.assertEqual(getattr(layout, runtime._LAYOUT_RANGES_ATTR), ((2, 14),))
        self.assertEqual(runtime._ranges_from_layout(layout), ((2, 14),))

        inside_existing = model._forward(
            None,
            None,
            None,
            transformer_options={"probe": runtime._AUDIO_RANGES.get},
            minimax_payload={"layout": layout},
        )
        inside_new = model._forward(
            None,
            None,
            None,
            transformer_options={"probe": runtime._AUDIO_RANGES.get},
            minimax_payload=None,
        )
        self.assertEqual(inside_existing, ((2, 14),))
        self.assertEqual(inside_new, ((2, 14),))
        self.assertEqual(runtime._AUDIO_RANGES.get(), ())

    def test_audio_ranges_fail_closed_when_invalid(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        self.assertEqual(runtime._normalize_ranges(((1, 3), (4, 6)), 6), ((1, 3), (4, 6)))
        self.assertEqual(runtime._normalize_ranges(((1, 7),), 6), ())
        self.assertEqual(runtime._normalize_ranges((("bad", 3),), 6), ())

    def test_install_is_idempotent(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        second = runtime.install_patch()
        self.assertTrue(second["installed"])
        self.assertEqual(second["version"], "0.1.2")
        self.assertEqual(second["reason"], "already installed")

    def test_conflicting_mlp_monkey_patch_disables_profile(self):
        model_module = self._install_fake_modules()

        def foreign_mlp_forward(self, x):
            return x

        foreign_mlp_forward.__module__ = "another_h3_extension"
        model_module.MLP.forward = foreign_mlp_forward
        runtime = self._load_runtime()
        self.assertFalse(runtime.PATCH_STATUS["installed"])
        self.assertIn("MLP.forward", runtime.PATCH_STATUS["reason"])

    def test_late_conflict_keeps_new_model_instances_unmodified(self):
        model_module = self._install_fake_modules()
        runtime = self._load_runtime()
        self.assertTrue(runtime.PATCH_STATUS["installed"])

        def foreign_mlp_forward(self, x):
            return x

        foreign_mlp_forward.__module__ = "another_h3_extension"
        model_module.MLP.forward = foreign_mlp_forward
        model = model_module.MiniMaxH3Model(num_layers=2)
        for block in model.blocks:
            self.assertFalse(hasattr(block.attn, runtime._ATTENTION_ENABLE_FLAG))

    def test_runtime_code_has_no_source_file_write_api(self):
        source = self.runtime_path.read_text(encoding="utf-8")
        for forbidden in ("open(", "write_text(", "write_bytes(", "replace(", "shutil.", "os.rename"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
