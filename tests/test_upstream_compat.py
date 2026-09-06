"""Run real upstream method bodies with trace tensors; no CUDA claim is made."""

import ast
from pathlib import Path
import sys
import types
import unittest

import test_runtime_patch as helpers


class _Schedule:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def __sub__(self, value):
        return _Schedule([x - value for x in self.values])

    def abs(self):
        return _Schedule([abs(x) for x in self.values])

    def argmin(self):
        return min(range(len(self.values)), key=self.values.__getitem__)

    def __getitem__(self, index):
        return self.values[index]


class UpstreamCompatibilityTests(unittest.TestCase):
    setUp = helpers.RuntimePatchTests.setUp
    tearDown = helpers.RuntimePatchTests.tearDown
    _install_fake_modules = helpers.RuntimePatchTests._install_fake_modules
    _load_runtime = helpers.RuntimePatchTests._load_runtime

    def load_upstream(self, filename):
        model, supported = self._install_fake_modules()
        path = Path(__file__).parent / "fixtures" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        namespace = dict(vars(model), torch=sys.modules["torch"])
        for name in ("_mod_row", "time_shift_sigma"):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        self.pdd_calls = []

        def pdd_head(head, h, n, start, stop, flow_shift):
            self.pdd_calls.append((head.name, h.dtype, n, start, stop, flow_shift))
            return head(h)

        namespace["_pdd_head"] = pdd_head
        for class_name, methods in {
            "Attention": ("forward",), "MLP": ("forward",),
            "DiTBlock": ("forward",), "FinalLayer": ("forward",),
            "MiniMaxH3Model": ("__init__", "_forward"),
            "PackedLayout": ("__init__",),
        }.items():
            source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            cls = type(class_name, (), {"__module__": model.__name__})
            for method in methods:
                function = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == method)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
                setattr(cls, method, namespace[method])
            setattr(model, class_name, cls)
        return model, supported, self._load_runtime()

    def final_inputs(self, runtime, banks=1, enabled=True):
        final = helpers._FakeFinalLayer()
        setattr(final, runtime._FINAL_ENABLE_FLAG, enabled)
        for head in (final.video_out, final.audio_out):
            head.out_features = 4
            head.weight.shape = (banks * 4, 16)
        trace = []
        return final, helpers._TraceTensor("x", "float16", trace), helpers._TraceTensor("t", "float16", trace), trace

    def test_v0345_real_source_installs_and_legacy_final_runs_fp32(self):
        _, _, runtime = self.load_upstream("model_v0345.py")
        self.assertTrue(runtime.PATCH_STATUS["installed"], runtime.PATCH_STATUS)
        final, x, t, trace = self.final_inputs(runtime)
        video, audio = runtime._patched_final_forward(final, x, t, (0, 4, 0), (4, 8, 1))
        self.assertEqual((video.dtype, audio.dtype), ("float32", "float32"))
        self.assertIn(("norm", "float32"), trace)
        self.assertIn(("t.to", "float32"), trace)
        self.assertIn(("video_out", "float32"), trace)
        self.assertIn(("audio_out", "float32"), trace)

    def test_pdd_real_source_installs_and_eight_positional_arguments_work(self):
        _, _, runtime = self.load_upstream("model_pdd.py")
        self.assertTrue(runtime.PATCH_STATUS["installed"], runtime.PATCH_STATUS)
        final, x, t, _ = self.final_inputs(runtime)
        video, audio = runtime._patched_final_forward(final, x, t, (0, 4, 0), (4, 8, 1), 1.0, None, (12.0, 3.0))
        self.assertEqual((video.dtype, audio.dtype), ("float32", "float32"))

    def test_pdd_bank_schedule_and_distinct_stream_shifts_are_preserved(self):
        _, _, runtime = self.load_upstream("model_pdd.py")
        final, x, t, trace = self.final_inputs(runtime, banks=4)
        # Video base-grid interval .0 -> .5 spans heads 0 and 1.
        schedule = _Schedule([1.0, 12.0 / 13.0, 0.0])
        runtime._patched_final_forward(final, x, t, (0, 4, 0), (4, 8, 1),
                                       sigma=1.0, sample_sigmas=schedule, shifts=(12.0, 3.0))
        self.assertEqual(self.pdd_calls, [("video_out", "float32", 4, 0, 2, 12.0),
                                         ("audio_out", "float32", 4, 0, 2, 3.0)])
        self.assertIn(("norm", "float32"), trace)

    def test_missing_pdd_schedule_still_raises_upstream_error(self):
        _, _, runtime = self.load_upstream("model_pdd.py")
        final, x, t, _ = self.final_inputs(runtime, banks=4)
        with self.assertRaisesRegex(ValueError, "sigma schedule"):
            runtime._patched_final_forward(final, x, t, (0, 4, 0), (4, 8, 1), 1.0, None, (12.0, 3.0))

    def test_inactive_cpu_and_training_paths_forward_arguments_without_casting(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        calls = []
        runtime._ORIGINAL_FINAL_FORWARD = lambda *args, **kwargs: calls.append((args, kwargs))
        for mode in ("inactive", "cpu", "training"):
            final, x, t, trace = self.final_inputs(runtime, enabled=mode != "inactive")
            if mode == "cpu":
                x.device.type = "cpu"
            runtime.model_management.in_training = mode == "training"
            schedule = object()
            runtime._patched_final_forward(final, x, t, (0, 4, 0), (4, 8, 1), 0.5,
                                           sample_sigmas=schedule, shifts=(12.0, 3.0))
            self.assertIs(calls[-1][0][1], x)
            self.assertIs(calls[-1][0][2], t)
            self.assertEqual(calls[-1][0][-1], 0.5)
            self.assertIs(calls[-1][1]["sample_sigmas"], schedule)
            self.assertEqual(trace, [])

    def test_final_forwards_future_keyword_and_segment_identity(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        runtime._ORIGINAL_FINAL_FORWARD = lambda *args, **kwargs: (args, kwargs)
        final, x, t, _ = self.final_inputs(runtime)
        video_seg, audio_seg, extension = object(), object(), object()
        args, kwargs = runtime._patched_final_forward(final, x, t, video_seg, audio_seg, extension=extension)
        self.assertIs(args[3], video_seg)
        self.assertIs(args[4], audio_seg)
        self.assertIs(kwargs["extension"], extension)
        self.assertEqual((args[1].dtype, args[2].dtype), ("float32", "float32"))

    def test_new_block_attention_override_receives_fp16_branch(self):
        self._install_fake_modules()
        runtime = self._load_runtime()
        block = helpers._FakeBlock()
        setattr(block, runtime._BLOCK_ENABLE_FLAG, True)
        block.mlp = lambda value: value
        calls = []

        def override(value, **kwargs):
            calls.append((value.dtype, kwargs))
            return value

        result = runtime._patched_block_forward(block, helpers._TraceTensor("x", "float16"),
            helpers._TraceTensor("t", "float16"), [(0, 8, 0)], "rope", {"test": True}, override)
        self.assertEqual(result.dtype, "float32")
        self.assertEqual(calls, [("float16", {"rope_freqs": "rope", "transformer_options": {"test": True}})])

    def test_inactive_new_block_delegates_attention_keyword(self):
        _, _, runtime = self.load_upstream("model_pdd.py")
        calls = []
        runtime._ORIGINAL_BLOCK_FORWARD = lambda *args, **kwargs: calls.append(kwargs)
        override = object()
        runtime._patched_block_forward(helpers._FakeBlock(), helpers._TraceTensor("x", "float16"),
                                       None, None, None, attention=override)
        self.assertIs(calls[0]["attention"], override)

    def test_unknown_block_parameter_disables_before_dtype_registration(self):
        model, supported = self._install_fake_modules()

        def future_block(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, new_feature=None):
            return x

        future_block.__module__ = model.__name__
        model.DiTBlock.forward = future_block
        runtime = self._load_runtime()
        self.assertFalse(runtime.PATCH_STATUS["installed"])
        self.assertIn("unknown parameters", runtime.PATCH_STATUS["reason"])
        self.assertEqual(supported.MiniMaxH3.supported_inference_dtypes, ["bfloat16", "float32"])


if __name__ == "__main__":
    unittest.main()
