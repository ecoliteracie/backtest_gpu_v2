# src/banners/phase6k.py
from __future__ import annotations
from .common import format_bytes

def build_banner(info: dict, sanity: dict) -> str:
    lines = []
    lines.append("=" * 53)
    lines.append("Phase 6K — GPU Backend Probe")
    lines.append("=" * 53)
    lines.append(f"[GPU] backend={info['backend']}, device={info['device_name']} (id={info['device_id']}), cc={info['cc']}")
    lines.append(f"[MEM] total={format_bytes(info['mem_total_bytes'])}, free={format_bytes(info['mem_free_bytes'])}")
    lines.append(f"[VER] driver={info['driver_ver']}, runtime={info['runtime_ver']}, cupy={info['cupy_version']}")
    lines.append(
        "[TEST] gpu_sum(sqrt(linspace(0,1,1024)))="
        f"{sanity['gpu_value']:.15f}, cpu={sanity['cpu_value']:.15f}, abs_err={sanity['abs_err']:.3e}"
    )
    lines.append("[LOG] Wrote: logs/phase06_gpu_probe.log")
    return "\n".join(lines)
