"""Build and verify the standalone v0.1.4 server-test ZIP (standard library)."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.4"
FOLDER = "minimax-h3-v100-l3-clean"
FILES = (
    "__init__.py", "runtime_patch.py", "README.md", "README_zh-CN.md",
    "NOTICE.md", "LICENSE", "tests/test_runtime_patch.py", "tests/test_upstream_compat.py",
    "tests/fixtures/model_v0345.py", "tests/fixtures/model_pdd.py",
    "tests/fixtures/PROVENANCE.json", "tools/build_release.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    payload = {name: (ROOT / name).read_bytes() for name in FILES}
    for name, data in payload.items():
        if name.endswith(".py"):
            ast.parse(data.decode("utf-8-sig"), filename=name)
    provenance = json.loads(payload["tests/fixtures/PROVENANCE.json"])
    for name, info in provenance["files"].items():
        assert digest(payload["tests/fixtures/" + name]) == info["sha256"], name
    for name in ("__init__.py", "runtime_patch.py"):
        assert f'"{VERSION}"' in payload[name].decode("utf-8"), name

    run = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                         cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if run.returncode:
        print(run.stdout + run.stderr)
        raise SystemExit(run.returncode)
    count = int(re.search(r"Ran (\d+) tests", run.stderr).group(1))
    manifest = {
        "package": "minimax-h3-v100-patch", "version": VERSION,
        "profile_id": "minimax-h3-v100-v014-native-fp16-branches",
        "install_folder": FOLDER, "generated_date": "2026-09-06",
        "compatibility": "ComfyUI v0.34.5 and PDD source fixtures; see tests/fixtures/PROVENANCE.json",
        "runtime_status": "V100 server inference, quality, speed and memory tests pending",
        "validation": {"python_ast": "passed", "dependency_free_tests": count,
                       "upstream_final_argument_and_dtype_flow": "passed",
                       "real_torch_numerical_tests": "not run", "gpu_inference": "not run"},
        "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()},
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (ROOT / "MANIFEST.json").write_bytes(manifest_bytes)
    payload["MANIFEST.json"] = manifest_bytes
    archive_path = ROOT.parent / f"minimax-h3-v100-v{VERSION}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(f"{FOLDER}/{name}", date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f"{FOLDER}/{name}" for name in payload}
        for name, data in payload.items():
            assert archive.read(f"{FOLDER}/{name}") == data, name
    archive_hash = digest(archive_path.read_bytes())
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
    report = f"{count} tests passed; Python AST and upstream source hashes passed; {len(payload)} ZIP entries verified.\n"
    (ROOT.parent / "v0.1.4-validation.txt").write_text(
        report + run.stderr + f"\nZIP SHA256: {archive_hash}\nGPU inference: not run; server test pending.\n",
        encoding="utf-8")
    print(report.strip())
    print(archive_path)
    print(archive_hash)


if __name__ == "__main__":
    main()
