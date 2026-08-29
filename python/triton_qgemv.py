"""The same quantized GEMV in Triton, plus a PyTorch reference.

Two reasons this file exists. Triton is named alongside CUDA in the job
descriptions this repository answers, and having both implementations lets the
same measurement be made twice through different compilers. The interesting
output is not which one wins, it is how far a few lines of Triton get you
against hand-written CUDA on a kernel that is supposed to be bandwidth bound.

The layout is the SoA one: a contiguous int8 quant plane and a separate fp16
scale plane. The ggml AoS block struct is 34 bytes and Triton has no good way
to address a 34-byte stride, which is itself part of the finding in the README.

Run:  python triton_qgemv.py --M 11008 --K 4096
"""

import argparse
import json
import statistics
import sys

import torch
import triton
import triton.language as tl

from qformats import (QK, dequant_q4_0, dequant_q8_0, quantize_q4_0,
                      quantize_q8_0)


# ------------------------------------------------------------------ kernels

@triton.jit
def _qgemv_q8_soa(
    Q,        # int8 [M, K]
    D,        # fp16 [M, K // QK]
    X,        # fp32 [K]
    Y,        # fp32 [M]
    M, K,
    BLOCK: tl.constexpr,
    QK: tl.constexpr,
):
    """One program per output row. BLOCK columns per iteration."""
    row = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        q = tl.load(Q + row * K + offs, mask=mask, other=0).to(tl.float32)
        x = tl.load(X + offs, mask=mask, other=0.0)
        # One scale per QK columns, gathered per element. The compiler folds
        # this to one load per block of 32 because the index is uniform there.
        d = tl.load(D + row * (K // QK) + offs // QK, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(q * d * x, axis=0)
    tl.store(Y + row, acc)


@triton.jit
def _qgemv_q4_soa(
    Q,        # uint8 [M, K // 2], ggml nibble order: low half then high half
    D,        # fp16 [M, K // QK]
    X,        # fp32 [K]
    Y,        # fp32 [M]
    M, K,
    NBLK: tl.constexpr,   # q4_0 blocks handled per iteration
    QK: tl.constexpr,
):
    row = tl.program_id(0)
    nb = K // QK
    acc = tl.zeros((), dtype=tl.float32)
    for b0 in range(0, nb, NBLK):
        blocks = b0 + tl.arange(0, NBLK)
        bmask = blocks < nb
        # Byte offsets inside each block: 16 bytes hold 32 weights.
        half = tl.arange(0, QK // 2)
        byte_off = blocks[:, None] * (QK // 2) + half[None, :]
        p = tl.load(Q + row * (K // 2) + byte_off, mask=bmask[:, None], other=0)
        lo = (p & 0x0F).to(tl.float32) - 8.0
        hi = ((p >> 4) & 0x0F).to(tl.float32) - 8.0
        x_lo = tl.load(X + blocks[:, None] * QK + half[None, :], mask=bmask[:, None], other=0.0)
        x_hi = tl.load(X + blocks[:, None] * QK + half[None, :] + 16, mask=bmask[:, None], other=0.0)
        d = tl.load(D + row * nb + blocks, mask=bmask, other=0.0).to(tl.float32)
        partial = tl.sum(lo * x_lo + hi * x_hi, axis=1)
        acc += tl.sum(partial * d, axis=0)
    tl.store(Y + row, acc)


# ------------------------------------------------------------------ timing

def time_ms(fn, warmup=20, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    samples.sort()
    return {
        "median": samples[len(samples) // 2],
        "min": samples[0],
        "p95": samples[int(0.95 * (len(samples) - 1))],
        "sd": statistics.pstdev(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=11008)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(1234)
    M, K = args.M, args.K
    W = (torch.rand(M, K, device=dev) * 2 - 1)
    x = (torch.rand(K, device=dev) * 2 - 1)

    p = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "device": p.name, "cc": f"sm_{p.major}{p.minor}",
        "sms": p.multi_processor_count, "torch": torch.__version__,
        "triton": triton.__version__, "M": M, "K": K,
    }))

    results = []

    # ---- q8_0
    q8, d8 = quantize_q8_0(W)
    ref8 = dequant_q8_0(q8, d8) @ x          # PyTorch reference, fp32
    y = torch.empty(M, device=dev, dtype=torch.float32)

    def run8():
        _qgemv_q8_soa[(M,)](q8, d8, x, y, M, K, BLOCK=args.block, QK=QK)

    run8()
    torch.cuda.synchronize()
    err8 = (y - ref8).abs().max().item()
    rel8 = err8 / ref8.abs().max().item()
    t8 = time_ms(run8, reps=args.reps)
    wb8 = M * K * 34 / 32                     # same byte count the CUDA harness uses
    results.append({"kernel": "triton_q8_soa", "format": "q8_0",
                    "ms_median": t8["median"], "ms_min": t8["min"], "ms_sd": t8["sd"],
                    "gbps": (wb8 / 1e9) / (t8["median"] / 1e3),
                    "max_abs_err": err8, "max_rel_err": rel8})

    # ---- q4_0
    q4, d4 = quantize_q4_0(W)
    ref4 = dequant_q4_0(q4, d4) @ x
    y4 = torch.empty(M, device=dev, dtype=torch.float32)

    def run4():
        _qgemv_q4_soa[(M,)](q4, d4, x, y4, M, K, NBLK=16, QK=QK)

    run4()
    torch.cuda.synchronize()
    err4 = (y4 - ref4).abs().max().item()
    rel4 = err4 / ref4.abs().max().item()
    t4 = time_ms(run4, reps=args.reps)
    wb4 = M * K * 18 / 32
    results.append({"kernel": "triton_q4_soa", "format": "q4_0",
                    "ms_median": t4["median"], "ms_min": t4["min"], "ms_sd": t4["sd"],
                    "gbps": (wb4 / 1e9) / (t4["median"] / 1e3),
                    "max_abs_err": err4, "max_rel_err": rel4})

    # ---- fp16 cuBLAS matvec, for scale. Not a competitor: it reads 2 bytes per
    # weight instead of 1.06 or 0.56, so it is here to show what the same shape
    # costs when nothing is quantized, and to sanity check the harness.
    Wh = W.to(torch.float16)
    xh = x.to(torch.float16)
    yh = torch.empty(M, device=dev, dtype=torch.float16)

    def runh():
        torch.mv(Wh, xh, out=yh)

    th = time_ms(runh, reps=args.reps)
    results.append({"kernel": "torch_fp16_mv", "format": "fp16",
                    "ms_median": th["median"], "ms_min": th["min"], "ms_sd": th["sd"],
                    "gbps": (M * K * 2 / 1e9) / (th["median"] / 1e3),
                    "max_abs_err": None, "max_rel_err": None})

    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
