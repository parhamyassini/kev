"""The accelerator this process uses: cuda, then mps, then cpu, or an Ascend npu when it is asked for by name."""
import subprocess
import sys
import warnings
from pathlib import Path

import torch

DEVICES = ("cpu", "cuda", "mps", "npu")


def default_device():
    return select()


def select(device="auto", *, dtype=torch.float32):
    """Resolve a device before loading weights. A fresh interpreter contains native driver crashes.

    Explicit CPU never inspects accelerators. Auto falls back on probe failure; an explicit accelerator fails closed.
    This tests one allocation/matmul/sync, not every model kernel or whether the checkpoint fits in memory.

    Auto considers cuda, mps and cpu. An Ascend NPU is asked for by name, because discovering one means importing
    torch_npu, which rewrites torch's dispatch for the rest of the process; ASCEND_RT_VISIBLE_DEVICES picks the card,
    as CUDA_VISIBLE_DEVICES does for cuda.
    """
    if device != "auto" and device not in DEVICES:
        raise ValueError(f"unsupported device {device!r}; choose auto or one of {DEVICES}")
    if device == "cpu":
        return device
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), device, str(dtype).removeprefix("torch.")],
                                capture_output=True, text=True, timeout=30, check=True)
        resolved = result.stdout.strip()
        if resolved not in DEVICES or device != "auto" and resolved != device:
            raise ValueError(f"unexpected probe result {resolved!r}")
        if resolved == "npu":
            import torch_npu  # noqa: F401  # registers torch.npu in this process too; must precede any .to("npu")
        return resolved
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError) and (stderr := (error.stderr or "").strip()):
            detail += f": {stderr.splitlines()[-1]}"
        message = f"Device probe failed ({detail})."
        if device != "auto":
            raise RuntimeError(f"{message} Fix the accelerator runtime or pass --device cpu.") from error
        warnings.warn(f"{message} Falling back to CPU; pass --device cpu to skip probing. "
                      "CPU defaults to fp32; KEV_DTYPE=bf16 reduces serving weight memory.", RuntimeWarning, stacklevel=2)
        return "cpu"


def _probe(device, dtype):
    """Child-process entry point: no accelerator operations from here may run in the selecting process."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "npu":
        import torch_npu  # noqa: F401  # the Ascend backend, in the child first: a broken runtime dies here
    if device != "cpu":
        with torch.no_grad():
            matrix = torch.ones(8, 8, device=device, dtype=dtype)
            matrix @ matrix
            sync(device)
    return device


def sync(device):
    """Wait for queued kernels, so wall-clock timings around a forward pass are real."""
    if device == "mps": torch.mps.synchronize()
    elif device == "cuda": torch.cuda.synchronize()
    elif str(device).startswith("npu"): torch.npu.synchronize()   # a tensor's device reads "npu:0"


def empty_cache(device):
    if device == "mps": torch.mps.empty_cache()
    elif device == "cuda": torch.cuda.empty_cache()
    elif str(device).startswith("npu"): torch.npu.empty_cache()


def out_of_memory(e):
    """Whether a torch allocator ran out: CUDA raises torch.OutOfMemoryError, MPS a plain RuntimeError with this message.
    The MLX backend's Metal errors are neither."""
    return isinstance(e, torch.OutOfMemoryError) or isinstance(e, RuntimeError) and str(e).startswith("MPS backend out of memory")


def allocated_bytes(device):
    """Bytes currently allocated on the device (MPS) or the peak since the process started (CUDA/NPU); 0 on CPU."""
    if device == "mps": return torch.mps.current_allocated_memory()
    if device == "cuda": return torch.cuda.max_memory_allocated()
    if str(device).startswith("npu"): return torch.npu.max_memory_allocated()
    return 0


if __name__ == "__main__":
    print(_probe(sys.argv[1], getattr(torch, sys.argv[2])))
