"""Startup selection, native-crash containment, and the actual serve argument path; no GPU or weights needed."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kev import device, serve
from kev.checkpoint import LoadOptions


def test_explicit_cpu_never_discovers_accelerators(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU selection must not inspect an accelerator or launch a probe")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.backends.mps, "is_available", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert device.select("cpu") == "cpu"


@pytest.mark.parametrize("requested,resolved", [("auto", "cuda"), ("auto", "mps"), ("auto", "cpu"),
                                               ("cuda", "cuda"), ("mps", "mps")])
def test_selection_uses_fresh_interpreter_and_requested_precision(monkeypatch, requested, resolved):
    def run(command, **kwargs):
        assert command == [sys.executable, str(Path(device.__file__).resolve()), requested, "bfloat16"]
        assert kwargs == dict(capture_output=True, text=True, timeout=30, check=True)
        return SimpleNamespace(stdout=resolved + "\n")
    monkeypatch.setattr(subprocess, "run", run)
    assert device.select(requested, dtype=torch.bfloat16) == resolved


@pytest.mark.parametrize("failure", [subprocess.CalledProcessError(1, "probe", stderr="HIP error: invalid device function"),
                                     subprocess.CalledProcessError(1, "probe", stderr=" \n"),
                                     subprocess.CalledProcessError(-11, "probe"), subprocess.TimeoutExpired("probe", 30),
                                     OSError("cannot launch interpreter")])
@pytest.mark.parametrize("requested", ["auto", "cuda", "mps", "npu"])
def test_failed_probe_falls_back_only_for_auto(monkeypatch, failure, requested):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "run", run)
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="Falling back to CPU.*KEV_DTYPE=bf16"):
            assert device.default_device() == "cpu"
    else:
        with pytest.raises(RuntimeError, match="--device cpu"):
            device.select(requested)


def test_invalid_probe_output_is_not_a_device(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="not a device\n"))
    with pytest.warns(RuntimeWarning, match="unexpected probe result"):
        assert device.select() == "cpu"
    with pytest.raises(ValueError, match="unsupported device"):
        device.select("tpu")


@pytest.mark.parametrize("cuda,mps,resolved", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_probe_allocates_multiplies_and_synchronizes(monkeypatch, cuda, mps, resolved):
    calls = []
    class Matrix:
        def __matmul__(self, other):
            calls.append("matmul")
            return self
    def ones(*shape, **kwargs):
        calls.append((shape, kwargs))
        return Matrix()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(device, "sync", lambda dev: calls.append(("sync", dev)))
    assert device._probe("auto", torch.bfloat16) == resolved
    assert calls == ([] if resolved == "cpu" else [((8, 8), {"device": resolved, "dtype": torch.bfloat16}),
                                                   "matmul", ("sync", resolved)])


@pytest.mark.skipif(os.name != "posix", reason="POSIX native signal exit status")
@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_real_sigsegv_in_probe_cannot_kill_parent(tmp_path, monkeypatch, requested):
    # Only the child imports this torch stand-in. It dies where a native allocation could die.
    fake_torch = '''
import contextlib, os, resource, signal
from types import SimpleNamespace
float32 = "float32"
cuda = SimpleNamespace(is_available=lambda: True)
no_grad = contextlib.nullcontext
def ones(*args, **kwargs):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGSEGV)
'''
    (tmp_path / "torch.py").write_text(fake_torch, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="SIGSEGV"):
            assert device.select() == "cpu"
    else:
        with pytest.raises(RuntimeError, match="SIGSEGV"):
            device.select(requested)


@pytest.fixture
def startup(monkeypatch):
    """Replace weight loading and the socket server, leaving serve.main, selection and option defaults intact."""
    loaded = []
    model = SimpleNamespace(hybrid=True, backend="torch", dtype="float32")
    class Checkpoint:
        def __init__(self, run):
            self.requested = self.path = run
        def load(self, dev, opts):
            loaded.append((dev, opts))
            return None, model
    monkeypatch.setattr(serve, "Checkpoint", Checkpoint)
    monkeypatch.setattr(serve, "Server", lambda ck, tok, model, dev: SimpleNamespace(truncate_states=False))
    monkeypatch.setattr(serve.app.state, "server", None, raising=False)
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: loaded.append(kwargs))
    for name in os.environ:
        if name.startswith("KEV_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(serve, "fused_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["kev.serve", "--run", "fixture/model", "--port", "8019"])
    return loaded


@pytest.mark.parametrize("requested,resolved,hip", [(None, "cuda", None), ("auto", "cpu", None),
                                                  ("cpu", "cpu", None), ("cuda", "cuda", "6.4"),
                                                  ("cuda", "cuda", None), ("mps", "mps", None)])
def test_serve_main_selects_device_and_defaults(startup, monkeypatch, requested, resolved, hip):
    if requested is not None:
        sys.argv += ["--device", requested]
    monkeypatch.setattr(torch.version, "hip", hip)
    def run(command, **kwargs):
        assert command[-2:] == [requested or "auto", "bfloat16"]
        assert requested != "cpu", "explicit CPU should never launch a probe"
        return SimpleNamespace(stdout=resolved)
    monkeypatch.setattr(subprocess, "run", run)
    serve.main()
    dev, opts = startup[0]
    assert dev == resolved
    assert opts.dtype == (None if dev == "cpu" else torch.bfloat16)
    assert opts.attn == ("sdpa" if dev == "mps" else None)
    assert opts.backend == "auto"
    assert opts.cuda_graphs == (True if dev == "cuda" and hip is None else None)
    assert opts.fused == (True if dev == "cuda" and hip is None else None)
    assert startup[1] == {"host": "127.0.0.1", "port": 8019}


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_serve_probe_failure_precedes_weight_loading(startup, monkeypatch, requested):
    sys.argv += ["--device", requested]
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(-11, "probe")
    monkeypatch.setattr(subprocess, "run", run)
    if requested == "auto":
        with pytest.warns(RuntimeWarning, match="Falling back to CPU"):
            serve.main()
        assert startup[0][0] == "cpu"
        assert startup[0][1].cuda_graphs is None and startup[0][1].fused is None
    else:
        with pytest.raises(RuntimeError, match="--device cpu"):
            serve.main()
        assert not startup


def test_serve_preserves_explicit_options_and_probes_that_dtype(startup, monkeypatch):
    sys.argv += ["--device", "cuda"]
    opts = LoadOptions(dtype=torch.float32, cuda_graphs=False, fused=False, backend="torch", attn="eager")
    monkeypatch.setattr(LoadOptions, "from_env", classmethod(lambda cls: opts))
    def run(command, **kwargs):
        assert command[-1] == "float32"
        return SimpleNamespace(stdout="cuda")
    monkeypatch.setattr(subprocess, "run", run)
    serve.main()
    assert startup[0] == ("cuda", opts)


def test_serve_rejects_invalid_device_before_loading(startup):
    sys.argv += ["--device", "bogus"]
    with pytest.raises(SystemExit) as error:
        serve.main()
    assert error.value.code == 2 and not startup


def test_serve_cpu_honors_bf16_environment_without_probe(startup, monkeypatch):
    sys.argv += ["--device", "cpu"]
    monkeypatch.setenv("KEV_DTYPE", "bf16")
    def forbidden(*args, **kwargs):
        pytest.fail("explicit CPU must skip probing")
    monkeypatch.setattr(subprocess, "run", forbidden)
    serve.main()
    assert startup[0][0] == "cpu"
    assert startup[0][1].dtype == torch.bfloat16


@pytest.fixture
def torch_npu_import(tmp_path, monkeypatch):
    """A torch_npu stand-in first on sys.path -> whether this process has imported it (the real one, if any, comes back)."""
    (tmp_path / "torch_npu.py").write_text("STAND_IN = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    real = sys.modules.pop("torch_npu", None)
    yield lambda: getattr(sys.modules.get("torch_npu"), "STAND_IN", False)
    sys.modules.pop("torch_npu", None)
    if real is not None:
        sys.modules["torch_npu"] = real


def test_npu_is_probed_in_a_child_before_this_process_imports_torch_npu(monkeypatch, torch_npu_import):
    """Auto never discovers an Ascend card. Asked for by name, the child probes it first, where a broken runtime dies,
    and only then does this process import torch_npu, which must precede any .to("npu")."""
    def run(command, **kwargs):
        assert command[-2:] == ["npu", "bfloat16"]
        assert not torch_npu_import(), "torch_npu must not load here before the probe passes"
        return SimpleNamespace(stdout="npu\n")
    monkeypatch.setattr(subprocess, "run", run)
    assert device.select("npu", dtype=torch.bfloat16) == "npu"
    assert torch_npu_import()


def test_failed_npu_probe_leaves_this_process_without_torch_npu(monkeypatch, torch_npu_import):
    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(-11, "probe")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="--device cpu"):
        device.select("npu")
    assert not torch_npu_import()


def test_probe_imports_torch_npu_then_multiplies_on_that_card(monkeypatch, torch_npu_import):
    calls = []
    class Matrix:
        def __matmul__(self, other):
            calls.append("matmul")
            return self
    monkeypatch.setattr(torch, "ones", lambda *shape, **kwargs: calls.append(kwargs["device"]) or Matrix())
    monkeypatch.setattr(device, "sync", lambda dev: calls.append(("sync", dev)))
    assert device._probe("npu", torch.bfloat16) == "npu"
    assert torch_npu_import() and calls == ["npu", "matmul", ("sync", "npu")]


def test_serve_main_on_an_npu_fuses_by_default_without_the_cuda_paths(startup, monkeypatch, torch_npu_import):
    sys.argv += ["--device", "npu"]
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(stdout=command[-2]))
    serve.main()
    dev, opts = startup[0]
    assert dev == "npu" and opts.dtype == torch.bfloat16
    assert opts.fused is True and opts.cuda_graphs is None


@pytest.mark.parametrize("module", ["kev.benchmark", "kev.experiment"])
def test_cli_help_resolves_no_device(monkeypatch, module):
    """--device defaults to auto and is resolved after parsing, so --help never launches a probe."""
    import importlib
    main = importlib.import_module(module).main
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("parsing arguments must not probe"))
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
