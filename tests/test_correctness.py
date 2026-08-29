"""Correctness gate. Fails the build if any kernel drifts from the reference.

The harness already checks every kernel against a double-precision CPU
reference over the same quantized bytes and reports the error. This turns that
report into a pass or fail with a stated tolerance, so that a regression is
noisy rather than a slightly larger number in a table nobody reads.

Tolerance: 1e-5 on the maximum relative error, against the largest element of
the reference vector. Measured worst case across nine kernels and two shapes is
4.5e-7, so the gate sits about twenty times above the observed noise. It is a
regression detector, not a claim that the kernels are accurate to 1e-5.
"""

import json
import os
import pathlib
import subprocess
import sys

TOL = 1e-5
BENCH = os.environ.get("BENCH", str(pathlib.Path.home() / "kgbuild" / "bench"))
SHAPES = [(4096, 4096), (11008, 4096), (2048, 1024)]


def run(M, K):
    out = subprocess.run(
        [BENCH, "--M", str(M), "--K", str(K), "--reps", "5"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"bench exited {out.returncode}\n{out.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            if "max_rel_err" in r:
                rows.append(r)
    return rows


def main():
    if not pathlib.Path(BENCH).exists():
        print(f"no bench binary at {BENCH}; run make first", file=sys.stderr)
        sys.exit(1)
    failures = []
    total = 0
    worst = 0.0
    for M, K in SHAPES:
        rows = run(M, K)
        if not rows:
            failures.append(f"{M}x{K}: harness produced no kernel records")
            continue
        for r in rows:
            total += 1
            worst = max(worst, r["max_rel_err"])
            status = "ok " if r["max_rel_err"] <= TOL else "FAIL"
            print(f"{status} {r['kernel']:16s} {M:6d}x{K:<6d} "
                  f"max_rel_err={r['max_rel_err']:.3g}")
            if r["max_rel_err"] > TOL:
                failures.append(f"{r['kernel']} at {M}x{K}: {r['max_rel_err']:.3g} > {TOL}")

    print(f"\n{total} kernel checks, worst relative error {worst:.3g}, tolerance {TOL}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all kernels match the double-precision reference")


if __name__ == "__main__":
    main()
