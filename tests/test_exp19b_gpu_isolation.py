"""GPU isolation between the exp19b draft and regeneration arms."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_pipeline():
    path = PROJECT_ROOT / "scripts" / "run_exp19b_pipeline.py"
    spec = importlib.util.spec_from_file_location("exp19b_pipeline_gpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_crossencoder_instantiation_in_selector_is_explicitly_cpu_only():
    source = (PROJECT_ROOT / "scripts" / "select_exp19b_evidence.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "CrossEncoder"]

    assert calls, "selector must instantiate its production CrossEncoder"
    for call in calls:
        device = [kw.value for kw in call.keywords if kw.arg == "device"]
        assert len(device) == 1
        assert isinstance(device[0], ast.Constant) and device[0].value == "cpu", \
            "every selector CrossEncoder must be pinned to device='cpu' even when CUDA exists"


def test_every_stage_between_draft_and_fingerprint_gate_hides_cuda():
    pipeline = _load_pipeline()
    stages = pipeline.build_stages(py="python")
    names = [name for name, _command in stages]
    between = stages[names.index("draft") + 1:names.index("draft_replay_check")]

    assert [name for name, _command in between] == ["extract", "select"]
    for name, command in between:
        assert command.env.get("CUDA_VISIBLE_DEVICES") == "", \
            f"{name} may not see CUDA between the draft and fingerprint gate"


def test_default_subprocess_runner_applies_the_cpu_only_environment(monkeypatch):
    pipeline = _load_pipeline()
    observed = []

    def fake_run(argv, env):
        observed.append((list(argv), env.copy()))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    code, _report = pipeline.run_pipeline(
        pipeline.build_stages(py="python"), fingerprint_fn=lambda: "stable",
        replay_check_fn=lambda: (True, "draft replay 5/5 bit-identical", {}),
        log=lambda _message: None)

    assert code == pipeline.EXIT_OK
    cpu_calls = [(argv, env) for argv, env in observed
                 if any(name in " ".join(argv)
                        for name in ("extract_exp19b_claims.py", "select_exp19b_evidence.py"))]
    assert len(cpu_calls) == 2
    assert all(env.get("CUDA_VISIBLE_DEVICES") == "" for _argv, env in cpu_calls)
