"""Energy per kernel, measured with the NVML monotonic counter.

Runs on the Windows host while the kernel runs under WSL, because it is the
same physical GPU either way and NVML is only reachable from the Windows side
on this machine.

Method note, and it is not incidental. A previous probe on this exact card
found that sampling nvmlDeviceGetPowerUsage perturbs the thing it measures:
unpolled idle draw was 0.233 W, polled at 20 Hz the counter read 4.297 W, an
extra 4.06 W on a 40 W part. So this script never samples. It reads
nvmlDeviceGetTotalEnergyConsumption exactly twice per window, once before and
once after, and takes the difference. Two reads over a twenty second window
cannot manufacture a watt.

Usage:
  py -3.11 energy_probe.py --sustain 20 --kernels q8_v5_soa_smem,q4_v5_soa_smem
"""

import argparse
import json
import re
import subprocess
import sys
import time

import pynvml

WSL_BENCH = "/home/manu/kgbuild/bench"


def counter_mj(h):
    """Monotonic energy in millijoules. One read, no sampling."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(h)


def measure_idle(h, seconds):
    e0 = counter_mj(h)
    t0 = time.perf_counter()
    time.sleep(seconds)
    e1 = counter_mj(h)
    t1 = time.perf_counter()
    dt = t1 - t0
    return {"seconds": dt, "joules": (e1 - e0) / 1000.0, "watts": (e1 - e0) / 1000.0 / dt}


def run_sustained(h, kernel, seconds, M, K):
    cmd = [
        "wsl.exe", "-d", "Ubuntu", "--", WSL_BENCH,
        "--M", str(M), "--K", str(K), "--sustain", str(seconds), "--only", kernel,
    ]
    e0 = counter_mj(h)
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds * 10 + 120)
    t1 = time.perf_counter()
    e1 = counter_mj(h)
    stdout = out.stdout.replace("\x00", "")
    rec = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"sustain"' in line:
            rec = json.loads(line)
    if rec is None:
        print("no sustain record; stdout was:\n" + stdout[:2000], file=sys.stderr)
        print("stderr:\n" + out.stderr[:2000], file=sys.stderr)
        return None
    wall = t1 - t0
    joules = (e1 - e0) / 1000.0
    return {
        "kernel": kernel,
        "M": M,
        "K": K,
        "wall_s": wall,
        "gpu_ms": rec["ms"],
        "iters": rec["iters"],
        "bytes_total": rec["bytes_total"],
        "joules_window": joules,
        "watts_window": joules / wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sustain", type=float, default=20.0)
    ap.add_argument("--idle", type=float, default=20.0)
    ap.add_argument("--M", type=int, default=11008)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--kernels", default="q8_v0_naive,q8_v1_warp,q8_v2_smem_x,"
                                          "q8_v3_soa_vec,q8_v5_soa_smem,"
                                          "q4_v0_naive,q4_v2_smem_x,q4_v5_soa_smem")
    ap.add_argument("--out", default="results/energy_rtx3050.json")
    args = ap.parse_args()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()

    out = {
        "device": name,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
        "method": "nvmlDeviceGetTotalEnergyConsumption, two reads per window, no sampling",
        "runs": [],
    }

    print("idle baseline (pre) ...", file=sys.stderr)
    out["idle_pre"] = measure_idle(h, args.idle)
    print(f"  {out['idle_pre']['watts']:.3f} W", file=sys.stderr)

    for k in args.kernels.split(","):
        k = k.strip()
        if not k:
            continue
        print(f"sustaining {k} for {args.sustain}s ...", file=sys.stderr)
        r = run_sustained(h, k, args.sustain, args.M, args.K)
        if r is None:
            continue
        idle_w = out["idle_pre"]["watts"]
        r["joules_above_idle"] = r["joules_window"] - idle_w * r["wall_s"]
        r["joules_per_iter"] = r["joules_window"] / r["iters"]
        r["pJ_per_weight_byte"] = r["joules_window"] / r["bytes_total"] * 1e12
        r["pJ_per_weight_byte_above_idle"] = (
            r["joules_above_idle"] / r["bytes_total"] * 1e12
        )
        out["runs"].append(r)
        print(f"  {r['watts_window']:.2f} W, {r['joules_per_iter']*1e3:.4f} mJ/call, "
              f"{r['pJ_per_weight_byte']:.2f} pJ/weight-byte", file=sys.stderr)

    print("idle baseline (post) ...", file=sys.stderr)
    out["idle_post"] = measure_idle(h, args.idle)
    print(f"  {out['idle_post']['watts']:.3f} W", file=sys.stderr)

    pynvml.nvmlShutdown()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
