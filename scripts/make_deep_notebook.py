"""One comprehensive notebook per GPU. Everything the study needs, in one run.

Beyond the quick notebook this adds:

  - a ceiling measured at several buffer sizes, not one
  - a working-set sweep from a quarter of L2 to sixteen times it, which turns
    the cache-residency problem into a measured curve instead of a caveat
  - three independent repeats of the headline shape, because a laptop run of
    the ALU-bound kernel moved 25% between sessions while its within-run
    spread was 0.5%, and that gap is the interesting number
  - a K sweep at fixed M
  - Nsight Compute over every kernel, not five
  - energy over every kernel
  - compute-sanitizer, which cannot attach on the development machine
  - the Triton and cuBLAS comparison

  python scripts/make_deep_notebook.py --target all
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = []

TARGETS = {
    "t4": ("Tesla T4", "T4"),
    "l4": ("NVIDIA L4", "L4"),
    "a100": ("A100", "A100"),
}


def _lines(src):
    return src.strip("\n").splitlines(keepends=True)


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "id": f"c{len(CELLS):02d}", "source": _lines(src)})


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "id": f"c{len(CELLS):02d}", "source": _lines(src)})


def build(target):
    expect, label = TARGETS[target]
    kernels = (ROOT / "src" / "qgemv_kernels.cuh").read_text(encoding="utf-8")
    bench = (ROOT / "src" / "bench.cu").read_text(encoding="utf-8")
    triton_src = (ROOT / "python" / "triton_qgemv.py").read_text(encoding="utf-8")

    md(f"""
# Quantized GEMV, comprehensive run: {label}

**Set the runtime first: Runtime, Change runtime type, {label}.** The next cell
checks what you actually got and names the output file after the real device,
so a substituted runtime cannot produce a mislabelled result. Two Kaggle
sessions that asked for a T4 both returned a P100, which is why this exists.

Expect roughly 20 to 40 minutes. Everything is measured in this session on the
device named below. Nothing is extrapolated.

What this run produces:

1. A bandwidth ceiling at several buffer sizes.
2. The full kernel ladder at every shape, including a **working-set sweep from
   a quarter of L2 to sixteen times it**. The first pass of this study measured
   cache bandwidth and called it memory bandwidth on the 48 MiB L2 parts; this
   sweep measures that transition rather than dodging it.
3. **Three independent repeats** of the headline shape. Within-run spread on
   the laptop was 0.5% while between-run spread on the ALU-bound kernel was
   25%, so the repeat is the honest error bar.
4. A K sweep at fixed M.
5. Nsight Compute counters for every kernel.
6. Energy per kernel from the NVML monotonic counter, read twice per window
   and never sampled.
7. compute-sanitizer, which cannot attach under WDDM on the dev machine.
8. Triton and cuBLAS for comparison.
""")

    code("""
import json, os, subprocess, sys, pathlib, re, time
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvcc','--version'], capture_output=True, text=True).stdout)
""")

    code(f"""
import torch
EXPECT = {expect!r}
actual = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
slug = re.sub(r'[^a-z0-9]+', '_', actual.lower()).strip('_')
OUT_NAME = f"deep_{{slug}}_results.json"
L2 = props.L2_cache_size
print("actual device :", actual, f"sm_{{cc[0]}}{{cc[1]}}")
print("SMs           :", props.multi_processor_count)
print("L2            :", f"{{L2/1048576:.1f}} MiB")
print("output file   :", OUT_NAME)
if EXPECT.lower() not in actual.lower():
    print()
    print("!" * 72)
    print(f"WRONG RUNTIME: targets {{EXPECT}}, got {{actual}}.")
    print("Runtime > Change runtime type > pick the right GPU > Run all.")
    print("The run still works and the file is named for the real device.")
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
arch = f"sm_{cc[0]}{cc[1]}"
print("building for", arch)
r = subprocess.run(
    ["nvcc", "-O3", "-std=c++17", f"-arch={arch}", "-lineinfo",
     "-I", str(SRC), str(SRC / "bench.cu"), "-o", str(SRC / "bench")],
    capture_output=True, text=True)
print("nvcc exit", r.returncode)
print(r.stdout[-2000:]); print(r.stderr[-2000:])
assert r.returncode == 0, "build failed"
""")

    code("""
results = {"device": actual, "cc": f"sm_{cc[0]}{cc[1]}", "l2_bytes": L2,
           "sms": props.multi_processor_count}

def run(args, quiet=False):
    r = subprocess.run([str(SRC / "bench")] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rec = json.loads(line)
            out.append(rec)
            if not quiet:
                print(json.dumps(rec))
    return out

K = 4096
BPR = K * 34 / 32                      # q8_0 bytes per row
free_b, _ = torch.cuda.mem_get_info()

def M_for_ratio(x):
    \"\"\"M such that the q8_0 weight matrix is x times the L2.\"\"\"
    m = int(x * L2 / BPR)
    return max(1024, (m // 32) * 32)

def host_ram():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024
    return 8 << 30

HOST_AVAIL = host_ram()
print(f"host RAM available: {HOST_AVAIL/2**30:.1f} GiB, device free: {free_b/2**30:.1f} GiB")

def fits(m):
    # The device holds several quantized copies. The host holds an fp32 matrix
    # four times the weight count, which is the binding constraint on the
    # smaller runtimes and is what actually OOMs if it is ignored.
    dev = m * BPR * 3.0
    host = m * K * 4 * 2.6
    return dev < 0.55 * free_b and host < 0.5 * HOST_AVAIL
""")

    md("""
## 1. The ceiling, at several sizes

One buffer size gives one number and no way to know whether it was the right
size. Three sizes that all agree is a ceiling; three that disagree means the
buffer was in cache.
""")

    code("""
results["ceiling"] = {}
for mb in [64, 256, 512, 1024]:
    if mb * 1048576 > 0.25 * free_b:
        continue
    print(f"--- {mb} MiB buffer ---")
    results["ceiling"][str(mb)] = run(["--mode", "ceiling", "--reps", "50",
                                       "--bytes", str(mb * 1048576)])
""")

    md("""
## 2. Working set against L2

Achieved bandwidth as the weight matrix grows from a quarter of L2 to sixteen
times it. The left end of this curve is cache bandwidth and the right end is
memory bandwidth, and the point of plotting it is that the study previously
reported a left-end number as if it were a right-end one.
""")

    code("""
results["l2_sweep"] = {}
ratios = [0.25, 0.5, 1, 2, 4, 8, 16]
for x in ratios:
    m = M_for_ratio(x)
    if not fits(m):
        print(f"skipping {x}x L2 (M={m}), does not fit in memory")
        continue
    reps = 200 if x <= 2 else 40
    tag = f"{x}xL2"
    print(f"=== working set {x}x L2, M={m}, weights={m*BPR/1048576:.0f} MiB, {reps} reps ===")
    results["l2_sweep"][tag] = {"M": m, "K": K, "ratio": x,
                                "rows": run([\"--M\", str(m), \"--K\", str(K),
                                             \"--reps\", str(reps)], quiet=True)}
    for r in results["l2_sweep"][tag]["rows"]:
        if r.get("kernel") in ("q8_v5_soa_smem", "q4_v5_soa_smem", "q8_v1_warp"):
            print(f"  {r['kernel']:18s} {r['ms_median']:9.4f} ms  {r['gbps']:8.1f} GB/s")
""")

    md("""
## 3. The canonical shapes, three times each

The headline numbers, repeated. The spread across these three is the error bar
that belongs in the writeup, not the within-run standard deviation, which on at
least one machine understated the real uncertainty by a factor of fifty.
""")

    code("""
results["repeats"] = {}
M_big = M_for_ratio(8)
if not fits(M_big):
    M_big = M_for_ratio(4)
CANON = [(4096, "4096x4096"), (11008, "11008x4096"), (M_big, f"{M_big}x{K}")]
for m, tag in CANON:
    results["repeats"][tag] = []
    reps = 200 if m <= 11008 else 40
    for i in range(3):
        rows = run(["--M", str(m), "--K", str(K), "--reps", str(reps)], quiet=True)
        results["repeats"][tag].append(rows)
        g = {r["kernel"]: r for r in rows if "gbps" in r}
        if "q8_v5_soa_smem" in g and "q4_v5_soa_smem" in g:
            print(f"{tag:16s} run {i+1}: q8 {g['q8_v5_soa_smem']['ms_median']:8.4f} ms, "
                  f"q4 {g['q4_v5_soa_smem']['ms_median']:8.4f} ms, "
                  f"ratio {g['q4_v5_soa_smem']['ms_median']/g['q8_v5_soa_smem']['ms_median']:.3f}")
    print()
results["M_big"] = M_big
""")

    md("""
## 4. K sweep

M fixed, K varied. Shared-memory staging holds K floats per block, so this is
where that strategy runs out of room.
""")

    code("""
results["k_sweep"] = {}
for kk in [1024, 2048, 4096, 8192, 16384]:
    m = 8192
    print(f"=== K={kk} (x tile = {kk*4/1024:.0f} KiB shared per block) ===")
    rows = run(["--M", str(m), "--K", str(kk), "--reps", "100"], quiet=True)
    results["k_sweep"][str(kk)] = rows
    for r in rows:
        if r.get("kernel") in ("q8_v5_soa_smem", "q4_v5_soa_smem"):
            print(f"  {r['kernel']:18s} {r['gbps']:8.1f} GB/s  relerr {r['max_rel_err']:.2g}")
""")

    md("""
## 5. Nsight Compute, every kernel

The counters that settle which pipe each kernel is waiting on. `ncu` flushes
caches, so these are unaffected by the residency question in section 2.
""")

    code("""
import glob, shutil
def find_ncu():
    p = shutil.which("ncu")
    if p: return p
    for pat in ("/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda*/bin/ncu"):
        hits = sorted(glob.glob(pat))
        if hits: return hits[-1]
    return None
ncu = find_ncu()
if ncu is None:
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True, text=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "nsight-compute"],
                   capture_output=True, text=True)
    ncu = find_ncu()
print("ncu:", ncu, "| uid", os.getuid())

ALL_KERNELS = ["q8_v0_naive", "q8_v1_warp", "q8_v2_smem_x", "q8_v3_soa_vec",
               "q8_v4_split2", "q8_v4_split4", "q8_v5_soa_smem",
               "q4_v0_naive", "q4_v1_warp", "q4_v2_smem_x", "q4_v3_soa_vec",
               "q4_v5_soa_smem", "q4_v6_fastunpack"]
METRICS = ",".join([
    "gpu__time_duration.avg", "dram__bytes_read.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "launch__occupancy_limit_shared_mem",
])
results["ncu"] = {}
if ncu:
    for k in ALL_KERNELS:
        r = subprocess.run(
            [ncu, "--metrics", METRICS, "--launch-count", "1",
             "--target-processes", "all", "--csv", str(SRC / "bench"),
             "--M", str(M_big), "--K", str(K), "--reps", "1", "--only", k],
            capture_output=True, text=True)
        blob = r.stdout + r.stderr
        if "ERR_NVGPUCTRPERM" in blob:
            print("counters refused on this VM. Nothing fixable from here.")
            break
        results["ncu"][k] = r.stdout
        vals = {}
        for line in r.stdout.splitlines():
            if line.startswith('"0"'):
                p = [q.strip('"') for q in re.findall(r'"([^"]*)"', line)]
                vals[p[-3]] = p[-1]
        if vals:
            print(f"{k:18s} dram {vals.get('dram__throughput.avg.pct_of_peak_sustained_elapsed','?'):>7}% "
                  f"sm {vals.get('sm__throughput.avg.pct_of_peak_sustained_elapsed','?'):>7}% "
                  f"sectors {vals.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','?'):>15} "
                  f"inst {vals.get('smsp__inst_executed.sum','?'):>15}")
""")

    md("""
## 6. compute-sanitizer

Cannot attach under WDDM on the development machine, so this is the only place
the kernels get a memory-safety check.
""")

    code("""
san = shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
r = subprocess.run([san, "--tool", "memcheck", str(SRC / "bench"),
                    "--M", "4096", "--K", "4096", "--reps", "2"],
                   capture_output=True, text=True)
tail = [l for l in (r.stdout + r.stderr).splitlines() if not l.strip().startswith("{")]
print("\\n".join(tail[-25:]))
results["sanitizer"] = "\\n".join(tail[-40:])
""")

    md("""
## 7. Energy, every kernel

`nvmlDeviceGetTotalEnergyConsumption` is a monotonic counter. It is read twice
per window and never sampled, because sampling `nvmlDeviceGetPowerUsage` at
20 Hz on one card in this study raised its measured idle draw from 0.233 W to
4.297 W. The counter needs Volta or newer; the cell says so rather than
substituting an estimate.
""")

    code("""
try:
    import pynvml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"],
                   capture_output=True, text=True)
    import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
try:
    pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    have = True
except pynvml.NVMLError as e:
    have = False
    print(f"{actual}: energy counter NOT supported ({e}); skipping energy")
    results["energy"] = {"supported": False, "device": actual}

if have:
    def window(fn):
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t0 = time.perf_counter()
        fn()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); t1 = time.perf_counter()
        return (e1 - e0) / 1000.0, t1 - t0

    j, s = window(lambda: time.sleep(20))
    idle_w = j / s
    print(f"idle {idle_w:.3f} W over {s:.1f} s")
    energy = {"supported": True, "device": actual, "idle_W": idle_w, "M": M_big, "runs": []}
    for k in ALL_KERNELS:
        rec = {}
        def go():
            rr = subprocess.run([str(SRC / "bench"), "--M", str(M_big), "--K", str(K),
                                 "--sustain", "15", "--only", k],
                                capture_output=True, text=True)
            for line in rr.stdout.splitlines():
                if '"sustain"' in line:
                    rec.update(json.loads(line))
        j, s = window(go)
        if not rec:
            print(k, "no sustain record"); continue
        row = {"kernel": k, "joules": j, "wall_s": s, "watts": j / s,
               "iters": rec["iters"], "bytes_total": rec["bytes_total"],
               "joules_above_idle": j - idle_w * s,
               "joules_per_iter": j / rec["iters"],
               "pJ_per_weight_byte": j / rec["bytes_total"] * 1e12}
        energy["runs"].append(row)
        print(f"{k:18s} {row['watts']:7.1f} W  {row['joules_per_iter']*1e3:9.4f} mJ/call  "
              f"{row['pJ_per_weight_byte']:8.1f} pJ/byte")
    results["energy"] = energy
    # idle again, because it moved by 1.3 W between windows on one machine
    j, s = window(lambda: time.sleep(20))
    results["energy"]["idle_W_post"] = j / s
    print(f"idle after {j/s:.3f} W")
""")

    md("""
## 8. Triton and cuBLAS

The same SoA kernel written in Triton, plus an unquantized cuBLAS matvec of the
same shape. cuBLAS is not a competitor here: it reads 2 bytes per weight rather
than 1.06, so it is the reference for what the memory system does when nothing
is quantized.
""")

    code("TRITON = r'''" + triton_src.replace("'''", "\\'\\'\\'") + "'''\n"
         "(SRC / 'triton_qgemv.py').write_text(TRITON)\n"
         "out = []\n"
         "for m in [11008, M_big]:\n"
         "    r = subprocess.run([sys.executable, str(SRC / 'triton_qgemv.py'),\n"
         "                        '--M', str(m), '--K', str(K), '--reps', '100'],\n"
         "                       capture_output=True, text=True)\n"
         "    print(r.stdout[-2500:] or r.stderr[-2000:])\n"
         "    out.append(r.stdout)\n"
         "results['triton'] = out")

    code("""
dest = pathlib.Path(OUT_NAME)
dest.write_text(json.dumps(results, indent=2))
print("wrote", dest, dest.stat().st_size, "bytes")
try:
    from google.colab import files
    files.download(str(dest))
except Exception as e:
    print("(download unavailable:", e, ")")
""")

    return {"cells": CELLS,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all")
    ap.add_argument("--outdir", default=str(ROOT / "notebooks"))
    a = ap.parse_args()
    targets = list(TARGETS) if a.target == "all" else [a.target]
    outdir = pathlib.Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        CELLS.clear()
        nb = build(t)
        p = outdir / f"qgemv_deep_{t}.ipynb"
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {p} ({len(nb['cells'])} cells, target={t})")
