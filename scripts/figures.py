"""Three figures, generated from results/. Nothing is drawn that was not measured.

  python scripts/figures.py

Type-3 fonts are disabled so the PDFs are conference-submittable.
"""

import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SHORT = {
    "NVIDIA GeForce RTX 3050 Laptop GPU": "RTX 3050",
    "Tesla P100-PCIE-16GB": "P100",
    "Tesla T4": "T4",
    "NVIDIA L4": "L4",
    "NVIDIA A100-SXM4-40GB": "A100",
}


def jsonl(name):
    p = RES / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip().startswith("{")]


def load_runs():
    """Every run, as (short device name, roof GB/s, {kernel: record})."""
    runs = []
    ceil = jsonl("ceiling_rtx3050.jsonl")
    lad = jsonl("ladder_98304x4096_rtx3050.jsonl") or jsonl("ladder_11008x4096_rtx3050.jsonl")
    if ceil and lad:
        dev = next(r for r in ceil if "device" in r)
        roof = next(r for r in ceil if r.get("kernel") == "stream_read")["gbps_median"]
        runs.append((SHORT.get(dev["device"], dev["device"]), roof,
                     {r["kernel"]: r for r in lad if "gbps" in r}, dev))
    for p in sorted(RES.glob("kaggle_*_results.json")) + sorted(RES.glob("colab_*_results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        tag = d.get("big_tag") or "11008x4096"
        if tag not in d:
            tag = "11008x4096"
        c = d["ceiling"]
        if isinstance(c, dict):
            c = max(c.values(), key=len)
        dev = next(r for r in c if "device" in r)
        roof = next(r for r in c if r.get("kernel") == "stream_read")["gbps_median"]
        runs.append((SHORT.get(dev["device"], dev["device"]), roof,
                     {r["kernel"]: r for r in d[tag] if "gbps" in r}, dev))
    return runs


def parse_ncu(blob):
    out = {}
    for line in blob.splitlines():
        if line.startswith('"0"'):
            p = [q.strip('"') for q in re.findall(r'"([^"]*)"', line)]
            if len(p) >= 3:
                out[p[-3]] = p[-1]
    return out


# ------------------------------------------------------------------ fig 1
def fig_ladder(runs):
    """The optimisation ladder as a fraction of each machine's own roof."""
    order = ["q8_v0_naive", "q8_v1_warp", "q8_v3_soa_vec", "q8_v2_smem_x", "q8_v5_soa_smem"]
    labels = ["naive\nthread/row", "warp/row\nAoS", "SoA\n16 B loads",
              "AoS\nx in shared", "SoA +\nx in shared"]
    # one line per device, averaging repeats
    agg = {}
    for name, roof, g, _ in runs:
        if not all(k in g for k in order):
            continue
        agg.setdefault(name, []).append([100 * g[k]["gbps"] / roof for k in order])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, series in sorted(agg.items(), key=lambda kv: -kv[1][0][-1]):
        ys = [sum(v[i] for v in series) / len(series) for i in range(len(order))]
        ax.plot(range(len(order)), ys, marker="o", label=f"{name} (n={len(series)})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("achieved bandwidth, % of that device's measured roof")
    ax.set_title("Quantized GEMV optimisation ladder, q8_0, 11008-98304 x 4096\n"
                 "each device normalised to its own measured streaming-read ceiling",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_ladder.{ext}", dpi=160)
    plt.close(fig)
    print("fig1_ladder")


# ------------------------------------------------------------------ fig 2
def fig_crossover(runs):
    """The conditional finding: q4_0 wins only once q8_0 is at the roof."""
    xs, ys, names = [], [], []
    for name, roof, g, _ in runs:
        if "q8_v5_soa_smem" not in g or "q4_v5_soa_smem" not in g:
            continue
        xs.append(100 * g["q8_v5_soa_smem"]["gbps"] / roof)
        ys.append(g["q4_v5_soa_smem"]["ms_median"] / g["q8_v5_soa_smem"]["ms_median"])
        names.append(name)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    seen = set()
    for x, y, n in zip(xs, ys, names):
        ax.scatter(x, y, s=70, zorder=3,
                   color="tab:green" if y < 1 else "tab:red",
                   edgecolor="black", linewidth=0.5)
        if n not in seen:      # one label per device; repeats are the spread
            seen.add(n)
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(8, -3),
                        fontsize=8)
    nrep = len(names) - len(seen)
    ax.text(0.02, 0.04, f"{len(names)} runs, {len(seen)} devices; repeat runs "
            f"plotted separately", transform=ax.transAxes, fontsize=7.5,
            color="0.35")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.text(0.02, 0.55, "q4_0 slower than q8_0", transform=ax.transAxes,
            fontsize=8, color="tab:red")
    ax.text(0.02, 0.46, "q4_0 faster than q8_0", transform=ax.transAxes,
            fontsize=8, color="tab:green")
    ax.set_xlabel("how close the q8_0 kernel gets to its own measured roof (%)")
    ax.set_ylabel("q4_0 time / q8_0 time")
    ax.set_title("Four-bit weights buy time only once you are bandwidth bound\n"
                 "q4_0 moves 47% fewer bytes and is still slower below the roof",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_format_crossover.{ext}", dpi=160)
    plt.close(fig)
    print("fig2_format_crossover")


# ------------------------------------------------------------------ fig 3
def fig_sectors():
    """The mechanism: identical weight traffic, wildly different load sectors."""
    order = ["q8_v1_warp", "q8_v3_soa_vec", "q8_v2_smem_x", "q8_v5_soa_smem"]
    labels = ["warp/row\nAoS", "SoA\n16 B", "AoS\nx shared", "SoA +\nx shared"]
    data = {}
    for p in sorted(RES.glob("colab_*_results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("ncu"):
            continue
        c = d["ceiling"]
        if isinstance(c, dict):
            c = max(c.values(), key=len)
        name = SHORT.get(next(r for r in c if "device" in r)["device"], "?")
        if name in data:
            continue
        vals = []
        for k in order:
            m = parse_ncu(d["ncu"].get(k, ""))
            v = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
            vals.append(float(v.replace(",", "")) if v else None)
        if all(v for v in vals):
            data[name] = vals
    if not data:
        print("no ncu data, skipping fig3")
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    w = 0.8 / len(data)
    for i, (name, vals) in enumerate(sorted(data.items())):
        ax.bar([x + i * w for x in range(len(order))], vals, width=w, label=name)
    ax.set_yscale("log")
    ax.set_xticks([x + 0.4 - w / 2 for x in range(len(order))])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("global load sectors (log scale)")
    ax.set_title("Same weight bytes, same arithmetic, 40x the load sectors\n"
                 "Nsight Compute, q8_0, the activation gather is the whole story",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_load_sectors.{ext}", dpi=160)
    plt.close(fig)
    print("fig3_load_sectors")


if __name__ == "__main__":
    runs = load_runs()
    print(f"{len(runs)} runs loaded")
    fig_ladder(runs)
    fig_crossover(runs)
    fig_sectors()
