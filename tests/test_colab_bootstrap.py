"""Exercise notebook setup in a running interpreter without editable paths."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("notebook_name", ["Inference", "LoRA"])
def test_setup_makes_source_importable_without_kernel_restart(tmp_path, notebook_name):
    project = Path(__file__).resolve().parents[1]
    work = tmp_path / "workspace"
    shutil.copytree(project / "src/datamate", work / "src/datamate",
                    ignore=shutil.ignore_patterns("__pycache__"))
    notebook = project / f"notebooks/DataMate_Colab_{notebook_name}.ipynb"
    driver = r'''
import importlib.util, json, sys, types
from pathlib import Path

notebook, work = map(Path, sys.argv[1:])
assert importlib.util.find_spec("datamate") is None, sys.path
print("Before setup: datamate is not importable")

# The pip/GPU steps are replaced; the DataMate import is real. This checks
# visibility in this process, without restarting Python or using site .pth files.
torch = types.ModuleType("torch")
torch.__version__ = "2.7.0+test"
torch.version = types.SimpleNamespace(cuda="test")
torch.cuda = types.SimpleNamespace(is_available=lambda: True)
sys.modules["torch"] = torch
for name in ("transformers", "peft"):
    module = types.ModuleType(name)
    module.__version__ = "test"
    module.AutoModelForImageTextToText = object
    sys.modules[name] = module
calls = []
def fake_run(command, **kwargs):
    calls.append((command, kwargs))
    if kwargs.get("stdout"):
        kwargs["stdout"].write("local-test-only\n")
    return types.SimpleNamespace(returncode=0)

save = work / "saved"
save.mkdir()
namespace = {"WORK": work, "SAVE": save, "sys": sys,
             "subprocess": types.SimpleNamespace(run=fake_run)}
cells = ["".join(cell["source"]) for cell in json.loads(
    notebook.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code"]
exec(compile(cells[1], str(notebook), "exec"), namespace)
from datamate.config import Settings
assert Settings().model_key == "qwen35-9b"
import datamate
assert Path(datamate.__file__).is_relative_to(work / "src")
pip_calls = [(command, kwargs) for command, kwargs in calls
             if "pip" in command and "install" in command]
assert pip_calls
assert all(kwargs["cwd"] == work and kwargs["check"] for _, kwargs in pip_calls)
assert (work / "colab-constraints.txt").read_text() == "torch==2.7.0+test\n"
print("After setup: DataMate import OK in the same Python process")
'''
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-X", "utf8", "-c", driver, str(notebook), str(work)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Before setup: datamate is not importable" in result.stdout
    assert "After setup: DataMate import OK" in result.stdout
