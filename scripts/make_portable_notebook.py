"""Generate a self-contained notebook that runs the identical kernel ladder on
whatever GPU Kaggle or Colab hands out.

The CUDA sources are read from src/ and embedded verbatim, so there is one
source of truth and the cloud run cannot silently drift from the local one.
The notebook prints the same JSON records the local harness prints, which is
what makes the platforms comparable at all.

  python scripts/make_portable_notebook.py --out notebooks/qgemv_portable.ipynb
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = []


def _lines(src):
    return src.strip("\n").splitlines(keepends=True)


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "id": f"c{len(CELLS):02d}", "source": _lines(src)})


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "id": f"c{len(CELLS):02d}", "source": _lines(src)})


TARGETS = {
    "t4":   ("Tesla T4",   "T4",   "sm_75"),
    "l4":   ("NVIDIA L4",  "L4",   "sm_89"),
    "a100": ("A100",       "A100", "sm_80"),
    "any":  (None,         "any",  None),
}


def build(target="any"):
    expect, label, arch_hint = TARGETS[target]
    kernels = (ROOT / "src" / "qgemv_kernels.cuh").read_text(encoding="utf-8")
    bench = (ROOT / "src" / "bench.cu").read_text(encoding="utf-8")
    triton_src = (ROOT / "python" / "triton_qgemv.py").read_text(encoding="utf-8")

    if expect:
        md(f"""
# Quantized GEMV kernel ladder: {label}

**Before running anything: Runtime, Change runtime type, and select the
{label} GPU.** Colab hands out whatever it has free unless told otherwise, and
this study has already been bitten by that: two Kaggle sessions that asked for a
T4 both returned a P100, and an older `out-t4` results directory turned out to
contain P100 data. The next cell checks what you actually got and says so
loudly.

One decode-path operation, `y = W x` with W quantized to ggml `q8_0` or `q4_0`,
written nine ways, identical sources to the RTX 3050 run. Results are written to
a filename derived from the **actual** device, never the expected one, so a
substituted runtime cannot produce a mislabelled file.

Every number printed below was measured in this session on the device this
notebook names. Nothing is extrapolated.
""")
    else:
        md("""
# Quantized GEMV: the same kernel ladder, on whatever GPU this is

One decode-path operation, `y = W x` with W quantized to ggml `q8_0` or `q4_0`,
written nine ways. The local run is an RTX 3050 Laptop (sm_86, 4 GB, GDDR6).
This notebook runs the identical sources on a datacenter part so the shape of
the optimisation ladder can be compared across memory systems.

Every number printed below was measured in this session on the device this
notebook names. Nothing is extrapolated.
""")

    code("""
import json, os, subprocess, sys, pathlib, textwrap, re
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvcc','--version'], capture_output=True, text=True).stdout)
""")

    code(f"""
# Which GPU did this session actually get? The output filename comes from this,
# not from what the notebook was named, so a substituted runtime is impossible
# to mislabel afterwards.
import torch
EXPECT = {expect!r}
actual = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
slug = re.sub(r'[^a-z0-9]+', '_', actual.lower()).strip('_')
OUT_NAME = f"colab_{{slug}}_results.json"
print("actual device :", actual, f"sm_{{cc[0]}}{{cc[1]}}")
print("output file   :", OUT_NAME)
if EXPECT and EXPECT.lower() not in actual.lower():
    print()
    print("!" * 72)
    print(f"WRONG RUNTIME: this notebook targets {{EXPECT}} but got {{actual}}.")
    print("Fix: Runtime > Change runtime type > pick the right GPU > Run all.")
    print("The run below will still work and the file will be named for the")
    print("device you actually got, so nothing gets mislabelled either way.")
    print("!" * 72)
else:
    print("runtime matches the target")
""")

    code("SRC = pathlib.Path('/tmp/qgemv'); SRC.mkdir(exist_ok=True)\n"
         "KERNELS = r'''" + kernels + "'''\n"
         "(SRC / 'qgemv_kernels.cuh').write_text(KERNELS)\n"
         "print('kernels', len(KERNELS), 'bytes')")

    code("BENCH = r'''" + bench + "'''\n"
         "(SRC / 'bench.cu').write_text(BENCH)\n"
         "print('bench', len(BENCH), 'bytes')")

    code("""
# Detect the architecture and build for it, rather than assuming one.
import torch
try:
    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}"
except Exception:
    arch = "sm_75"
print("building for", arch)
r = subprocess.run(
    ["nvcc", "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo",
     "-I", str(SRC), str(SRC / "bench.cu"), "-o", str(SRC / "bench")],
    capture_output=True, text=True)
print(r.returncode)
print(r.stdout[-3000:]); print(r.stderr[-3000:])
""")

    code("""
def run(args):
    r = subprocess.run([str(SRC / "bench")] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rec = json.loads(line)
            out.append(rec)
            print(json.dumps(rec))
    return out

results = {}
print("=== measured bandwidth ceiling ===")
results["ceiling"] = run(["--mode", "ceiling", "--reps", "50"])
print()

# Pick a shape whose weight matrix is far larger than this L2, or the timing
# loop measures cache bandwidth and calls it memory bandwidth. The first pass
# of this study got that wrong on the 48 MiB L2 parts and reported a kernel
# running at 274% of its own measured DRAM roof, which is impossible and was
# the tell. K is fixed at 4096; M is whatever makes the weights 8x the L2.
K = 4096
l2 = torch.cuda.get_device_properties(0).L2_cache_size
bytes_per_row_q8 = K * 34 / 32
M_big = int(8 * l2 / bytes_per_row_q8)
M_big = max(11008, (M_big // 32) * 32)
free_b, total_b = torch.cuda.mem_get_info()
# The harness holds q8 AoS + q4 AoS + both SoA planes at once, about 2.7x the
# q8 weight bytes, so keep well inside what is free.
while M_big > 11008 and M_big * bytes_per_row_q8 * 2.7 > 0.5 * free_b:
    M_big = (M_big // 2 // 32) * 32
print(f"L2 = {l2/1048576:.0f} MiB, free = {free_b/2**30:.1f} GiB")
print(f"large shape M = {M_big}, q8_0 weights = "
      f"{M_big*bytes_per_row_q8/1048576:.0f} MiB = "
      f"{M_big*bytes_per_row_q8/l2:.1f}x L2")

SHAPES = [(4096, K, "4096x4096"), (11008, K, "11008x4096")]
if M_big > 11008:
    SHAPES.append((M_big, K, f"{M_big}x{K}"))
results["shape_choice"] = {"l2_bytes": l2, "M_big": M_big, "K": K}

for M, Kk, tag in SHAPES:
    reps = 200 if M <= 11008 else 40
    print(f"=== ladder {tag} ({reps} reps) ===")
    results[tag] = run(["--M", str(M), "--K", str(Kk), "--reps", str(reps)])
    print()
results["big_tag"] = SHAPES[-1][2]
""")

    md("""
## Nsight Compute

This is the step that fails on the development machine, where `ncu` returns
`ERR_NVGPUCTRPERM` under WSL2.

Being root is **not** what fixes it. Kaggle has `ncu` installed and runs the
notebook as uid 0, and counters are refused there anyway, because the driver
gates counter access independently of the user and that Kaggle vGPU does not
pass it through to the guest. Colab does grant it. So this cell is the one place
in the study where the limiter is measured rather than inferred, and it only
works on one of the two free platforms.

The cell below locates `ncu`, installs Nsight Compute if the image does not ship
it, and reports plainly if counters are still refused rather than printing an
empty table.
""")

    code("""
import glob, shutil

def find_ncu():
    p = shutil.which("ncu")
    if p:
        return p
    for pat in ("/opt/nvidia/nsight-compute/*/ncu",
                "/usr/local/cuda*/bin/ncu",
                "/usr/local/NVIDIA-Nsight-Compute*/ncu"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None

ncu = find_ncu()
if ncu is None:
    print("ncu not present, installing nsight-compute ...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True, text=True)
    r = subprocess.run(["apt-get", "install", "-y", "-qq", "nsight-compute"],
                       capture_output=True, text=True)
    print("apt exit", r.returncode, r.stderr[-500:])
    ncu = find_ncu()
print("ncu:", ncu)
print("running as uid", os.getuid())
""")

    code("""
ncu_metrics = ",".join([
    "gpu__time_duration.avg",
    "dram__bytes_read.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
])

ncu_out = {}
if ncu is None:
    print("no ncu available on this image; skipping the profiler section")
else:
    Mp = results["shape_choice"]["M_big"]
    for k in ["q8_v1_warp", "q8_v3_soa_vec", "q8_v2_smem_x",
              "q8_v5_soa_smem", "q4_v5_soa_smem", "q4_v6_fastunpack"]:
        print("=" * 24, k)
        r = subprocess.run(
            [ncu, "--metrics", ncu_metrics, "--launch-count", "1",
             "--target-processes", "all", "--csv", str(SRC / "bench"),
             "--M", str(Mp), "--K", "4096", "--reps", "1", "--only", k],
            capture_output=True, text=True)
        blob = r.stdout + r.stderr
        if "ERR_NVGPUCTRPERM" in blob:
            print("counters refused on this VM (ERR_NVGPUCTRPERM). "
                  "Nothing to fix from inside the notebook.")
            break
        ncu_out[k] = r.stdout
        print(r.stdout[-3500:] or r.stderr[-1500:])
results["ncu"] = ncu_out
""")

    md("""
### Reading the profiler against the prediction

The controlled experiment on the development machine concluded that the
activation gather, not the weight stream, was the limiter for `q8_v1_warp` and
`q8_v3_soa_vec`, and that `q4_v5_soa_smem` is ALU bound rather than memory
bound. Those are falsifiable against the counters above:

- `q8_v1_warp` and `q8_v3_soa_vec` should show **low** `dram__throughput` with
  **high** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, which is the
  signature of a gather that never reaches DRAM efficiently.
- `q8_v5_soa_smem` should show the **highest** `dram__throughput` of the set.
- `q4_v5_soa_smem` should show high `sm__throughput` and `smsp__inst_executed`
  against modest `dram__throughput`, which is what "the unpacking cost more than
  the bytes saved" looks like in counters.

If those do not hold, the mechanism in the README is wrong and the README needs
changing, not the data.
""")

    md("""
## Energy

`nvmlDeviceGetTotalEnergyConsumption` is a monotonic counter, present on Volta
and newer. It is absent on the P100 (Pascal), and this cell says so rather than
substituting an integrated power estimate without labelling it. The counter is
read twice per window and never sampled, because sampling perturbs the draw on
at least one part in this study.
""")

    code("""
try:
    import pynvml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"],
                   capture_output=True, text=True)
    import pynvml
import time
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(h)
name = name.decode() if isinstance(name, bytes) else name
has_counter = True
try:
    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
except pynvml.NVMLError as e:
    has_counter = False
    print(f"{name}: energy counter NOT supported ({e}). Skipping energy.")

if has_counter:
    def window(fn):
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t0 = time.perf_counter()
        fn()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t1 = time.perf_counter()
        return (e1 - e0) / 1000.0, t1 - t0

    j_idle, s_idle = window(lambda: time.sleep(15))
    idle_w = j_idle / s_idle
    print(f"{name} idle {idle_w:.3f} W over {s_idle:.1f} s")
    energy = {"device": name, "idle_W": idle_w, "runs": []}
    Me = results["shape_choice"]["M_big"]
    for k in ["q8_v0_naive", "q8_v2_smem_x", "q8_v5_soa_smem",
              "q4_v5_soa_smem", "q4_v6_fastunpack"]:
        rec = {}
        def go():
            r = subprocess.run([str(SRC / "bench"), "--M", str(Me), "--K", "4096",
                                "--sustain", "20", "--only", k],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if '"sustain"' in line:
                    rec.update(json.loads(line))
        j, s = window(go)
        if not rec:
            print(k, "no sustain record"); continue
        row = {"kernel": k, "joules": j, "wall_s": s, "watts": j / s,
               "iters": rec["iters"], "bytes_total": rec["bytes_total"],
               "joules_above_idle": j - idle_w * s,
               "pJ_per_weight_byte": j / rec["bytes_total"] * 1e12}
        energy["runs"].append(row)
        print(json.dumps(row))
    results["energy"] = energy
""")

    code("TRITON = r'''" + triton_src.replace("'''", "\\'\\'\\'") + "'''\n"
         "(SRC / 'triton_qgemv.py').write_text(TRITON)\n"
         "r = subprocess.run([sys.executable, str(SRC / 'triton_qgemv.py'),\n"
         "                    '--M','11008','--K','4096'], capture_output=True, text=True)\n"
         "print(r.stdout[-4000:]); print(r.stderr[-3000:])")

    code("""
out = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path(".")
dest = out / OUT_NAME
dest.write_text(json.dumps(results, indent=2))
print("wrote", dest, dest.stat().st_size, "bytes")

# On Colab, hand the file straight back rather than leaving it in a VM that is
# about to be reclaimed.
try:
    from google.colab import files
    files.download(str(dest))
except Exception as e:
    print("(not Colab, or download unavailable:", e, ")")
""")

    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all",
                    help="t4, l4, a100, any, or all")
    ap.add_argument("--outdir", default=str(ROOT / "notebooks"))
    a = ap.parse_args()

    targets = ["t4", "l4", "a100", "any"] if a.target == "all" else [a.target]
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        CELLS.clear()
        nb = build(t)
        name = "qgemv_portable.ipynb" if t == "any" else f"qgemv_colab_{t}.ipynb"
        p = outdir / name
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p} ({len(nb['cells'])} cells, target={t})")
