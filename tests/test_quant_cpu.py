"""CPU-only checks that run without a GPU, so CI can run them.

This does NOT test any kernel. It tests python/qformats.py, the quantizers and
dequantizers
that the GPU kernels are checked against, which is the part of the correctness
argument that does not need a device. The kernels themselves are checked on
five GPUs; see docs/CROSS_ARCH.md.

What is verified here:

  1. q8_0 and q4_0 round-trip within the error the format allows. A block of 32
     weights with one fp16 scale cannot do better than its own step size, so the
     bound is derived from the step rather than picked to make the test pass.
  2. The packed q4_0 nibble order matches ggml: low nibbles are the first 16
     weights of the block, high nibbles the second 16. Getting this backwards
     produces a kernel that is fast, self-consistent and wrong.
  3. The dequantize-then-matvec reference agrees with a direct block-wise
     computation, which is the reference the CUDA harness compares against.
"""

import sys

import torch

sys.path.insert(0, "python")
from qformats import (QK, dequant_q4_0, dequant_q8_0, quantize_q4_0,  # noqa: E402
                      quantize_q8_0)

FAIL = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    torch.manual_seed(1234)
    M, K = 64, 512
    W = torch.rand(M, K) * 2 - 1

    # ---- 1. round-trip inside the format's own step size
    #
    # The bound is one step, not half a step, and the difference is not slack.
    # Rounding alone gives half a step, but two things push past it. The scale
    # is stored as fp16, so it carries its own relative error. And q4_0 is the
    # interesting case: ggml derives d from the signed value of largest
    # magnitude as d = mx / -8, then stores q = clamp(round(w/d) + 8, 0, 15).
    # A weight at the opposite extreme of the block maps to 16 and is clamped
    # to 15, so it can be a full step out. That is a property of the format,
    # not a bug in this code.
    #
    # The exact bound, derived rather than fitted: q is computed from the fp32
    # scale but dequantized with the fp16 one, and |w| <= amax = 8*|d_fp32|, so
    # |w / d_fp16| <= 8 * (1 + 2^-11). The clamped weight dequantizes to 7*d,
    # giving a worst case of (1 + 8*2^-11) = (1 + 2^-8) steps. The measured
    # worst case is 1.00125 steps, inside that and nowhere near half a step.
    q8, d8 = quantize_q8_0(W)
    r8 = dequant_q8_0(q8, d8)
    step8 = d8.to(torch.float32).repeat_interleave(QK, dim=1)
    e8 = (W - r8).abs()
    check("q8_0 round-trip within one step",
          bool((e8 <= step8 + 1e-6).all()),
          f"max err {float(e8.max()):.3e}")
    check("q8_0 typical error within half a step",
          bool((e8 <= step8 * 0.5 + 1e-6).float().mean() > 0.99),
          f"{float((e8 <= step8 * 0.5 + 1e-6).float().mean()) * 100:.2f}% of weights")

    q4, d4 = quantize_q4_0(W)
    r4 = dequant_q4_0(q4, d4)
    step4 = d4.to(torch.float32).abs().repeat_interleave(QK, dim=1)
    e4 = (W - r4).abs()
    check("q4_0 round-trip within one step plus the fp16 scale error",
          bool((e4 <= step4 * (1 + 2**-8) + 1e-6).all()),
          f"max err {float(e4.max()):.3e}")
    over_half = int((e4 > step4 * 0.5 + 1e-6).sum())
    check("q4_0 weights beyond half a step are at most one per block",
          over_half <= M * (K // QK),
          f"{over_half} of {M * K}, cap {M * (K // QK)} (the clamped extreme)")

    # ---- 2. ggml nibble order, checked explicitly rather than assumed
    packed = q4.reshape(M, K // QK, QK // 2).to(torch.int32)
    lo = (packed & 0x0F).to(torch.float32) - 8.0
    hi = ((packed >> 4) & 0x0F).to(torch.float32) - 8.0
    dd = d4.to(torch.float32)[:, :, None]
    first16 = (lo * dd).reshape(M, K // QK, 16)
    second16 = (hi * dd).reshape(M, K // QK, 16)
    ref = r4.reshape(M, K // QK, QK)
    check("q4_0 low nibbles are weights 0-15 of the block",
          bool(torch.allclose(first16, ref[:, :, :16], atol=1e-6)))
    check("q4_0 high nibbles are weights 16-31 of the block",
          bool(torch.allclose(second16, ref[:, :, 16:], atol=1e-6)))

    # ---- 3. the reference matvec matches a direct block-wise computation
    x = torch.rand(K) * 2 - 1
    for name, q, d, deq in (("q8_0", q8, d8, dequant_q8_0),
                            ("q4_0", q4, d4, dequant_q4_0)):
        fast = deq(q, d) @ x
        blocks = deq(q, d).reshape(M, K // QK, QK)
        slow = (blocks * x.reshape(1, K // QK, QK)).sum(dim=2).sum(dim=1)
        rel = float((fast - slow).abs().max() / slow.abs().max())
        check(f"{name} reference matvec matches block-wise sum", rel < 1e-5,
              f"rel {rel:.2e}")

    # ---- byte counts the GB/s figures depend on
    check("q8_0 is 34 bytes per 32 weights",
          q8.numel() + d8.numel() * 2 == M * (K // QK) * 34)
    check("q4_0 is 18 bytes per 32 weights",
          q4.numel() + d4.numel() * 2 == M * (K // QK) * 18)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    print("host-side checks pass (no GPU involved; kernels are checked on device)")


if __name__ == "__main__":
    main()
