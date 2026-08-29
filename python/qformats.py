"""ggml q8_0 and q4_0 block formats, in pure PyTorch.

Separated from the Triton kernels deliberately: nothing here needs a GPU or a
Triton install, so the format code can be imported and tested on a CPU-only
runner. That is the difference between CI that checks something and CI that
checks that the imports resolve.

The layouts match ggml byte for byte, which is what lets a row of W here be a
row of W in a GGUF file.
"""

import torch

QK = 32


def quantize_q8_0(W):
    """ggml q8_0: per 32-weight block, d = amax/127, q = round(w/d)."""
    M, K = W.shape
    Wb = W.reshape(M, K // QK, QK)
    amax = Wb.abs().amax(dim=2)
    d = amax / 127.0
    idv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    q = torch.round(Wb * idv[:, :, None]).clamp(-127, 127).to(torch.int8)
    return q.reshape(M, K).contiguous(), d.to(torch.float16).contiguous()


def quantize_q4_0(W):
    """ggml q4_0: d = (signed value of largest magnitude) / -8, q = round(w/d)+8."""
    M, K = W.shape
    Wb = W.reshape(M, K // QK, QK)
    idx = Wb.abs().argmax(dim=2, keepdim=True)
    mx = torch.gather(Wb, 2, idx).squeeze(2)
    d = mx / -8.0
    idv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp((Wb * idv[:, :, None] + 8.5).to(torch.int32), 0, 15)
    lo = q[:, :, :16]
    hi = q[:, :, 16:]
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed.reshape(M, K // 2).contiguous(), d.to(torch.float16).contiguous()


def dequant_q8_0(q, d):
    M, K = q.shape
    return (q.reshape(M, K // QK, QK).to(torch.float32)
            * d.to(torch.float32)[:, :, None]).reshape(M, K)


def dequant_q4_0(packed, d):
    M, Kh = packed.shape
    K = Kh * 2
    p = packed.reshape(M, K // QK, QK // 2).to(torch.int32)
    lo = (p & 0x0F).to(torch.float32) - 8.0
    hi = ((p >> 4) & 0x0F).to(torch.float32) - 8.0
    w = torch.cat([lo, hi], dim=2)
    return (w * d.to(torch.float32)[:, :, None]).reshape(M, K)


