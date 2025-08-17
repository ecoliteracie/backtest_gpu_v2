# src/gpu_backend.py
from __future__ import annotations

from typing import Any, Dict


def format_bytes(n: int) -> str:
    """
    Render bytes as a human-readable binary size (MiB/GiB).
    """
    if n is None:
        return "n/a"
    step = 1024.0
    g = n / (step * step * step)
    if g >= 1.0:
        return f"{g:.1f} GiB"
    m = n / (step * step)
    return f"{m:.1f} MiB"


def sanity_compute_check(cp) -> dict:
    """
    Perform a small deterministic compute fully on GPU and cross-check on CPU.
    Uses float64 throughout to ensure tight tolerance.
    Returns dict with gpu_value, cpu_value, abs_err.
    """
    import numpy as np

    x_gpu = cp.linspace(0.0, 1.0, 1024, dtype=cp.float64)
    y_gpu = cp.sqrt(x_gpu).sum()
    gpu_val = float(y_gpu.get())  # bring to host

    x_cpu = np.linspace(0.0, 1.0, 1024, dtype=np.float64)
    cpu_val = float((x_cpu**0.5).sum())

    abs_err = abs(gpu_val - cpu_val)
    return {"gpu_value": gpu_val, "cpu_value": cpu_val, "abs_err": abs_err}


def select_backend(preferred: str = "cupy") -> Dict[str, Any]:
    """
    Probe the GPU backend (CuPy) and return device/runtime info.
    On failure, raises SystemExit with a concise, actionable message.
    """
    if preferred != "cupy":
        raise SystemExit("GPU ERROR: Only 'cupy' backend is supported in Phase 6K.")

    # Try to import CuPy
    try:
        import cupy as cp  # type: ignore
    except Exception:
        raise SystemExit("GPU ERROR: CuPy not installed or import failed. Install a CUDA-compatible CuPy build.")

    # Basic device handshake
    try:
        dev_count = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise SystemExit(f"GPU ERROR: Unable to query CUDA devices: {e}")

    if dev_count < 1:
        raise SystemExit("GPU ERROR: No CUDA device detected. Ensure NVIDIA driver + CUDA runtime are installed.")

    try:
        dev = cp.cuda.Device()  # current device
        dev_id = int(dev.id)
    except Exception as e:
        raise SystemExit(f"GPU ERROR: Unable to acquire active CUDA device: {e}")

    # Query props, mem, versions
    try:
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name_raw = props.get("name", b"")
        if isinstance(name_raw, (bytes, bytearray)):
            device_name = name_raw.decode(errors="replace")
        else:
            device_name = str(name_raw)

        major = int(props.get("major", -1))
        minor = int(props.get("minor", -1))
        cc = f"{major}.{minor}"

        free_b, total_b = cp.cuda.runtime.memGetInfo()
        driver_ver = cp.cuda.runtime.driverGetVersion()
        runtime_ver = cp.cuda.runtime.runtimeGetVersion()
        cupy_version = getattr(cp, "__version__", "unknown")
    except Exception as e:
        raise SystemExit(f"GPU ERROR: Unable to query device/runtime properties: {e}")

    # Tiny allocation + compute test (on device)
    try:
        _x = cp.arange(8, dtype=cp.float64)
        _y = (2.0 * _x).sum()
        _ = float(_y.get())
    except Exception as e:
        raise SystemExit(f"GPU ERROR: Device allocation/compute failed: {e}")

    return {
        "backend": "cupy",
        "gpu_available": True,
        "device_id": dev_id,
        "device_name": device_name,
        "cc": cc,
        "mem_total_bytes": int(total_b),
        "mem_free_bytes": int(free_b),
        "driver_ver": int(driver_ver),
        "runtime_ver": int(runtime_ver),
        "cupy_version": cupy_version,
        "xp": cp,  # pass CuPy module forward for later phases
    }
