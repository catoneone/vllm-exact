"""Linear layers with exact INT8 two-digit accumulation: digitization + matmul (Triton kernels and torch references).

Numerical spec:
- Activations and weights are converted per row to F-bit fixed point (one power-of-two scale per row) and split into
  two INT8 digits q = 256*h + l. F = 15 uses offset digits (with a row-sum correction), F = 14 uses balanced digits;
  trunc=True drops the l*l term (3 INT8 dots instead of 4).
- Products are exact in INT8 tensor cores and accumulation is in int32 / int64. Integer addition is exact and
  order-independent, so the result does not depend on tiling, split-K partitioning, batch size, or the other rows in
  the batch. Finally int64 -> fp64 (exact) -> fp32 (one rounding), then bf16 if requested.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = torch.cuda.is_available()
except Exception:  # CPU-only machine
    HAVE_TRITON = False


# ======================= fixed-point digitization =======================
def digitize_torch(x, F):
    """x: (R, K) fp32 -> (h int8, l int8, cs int32 row-sum correction, sh int32). Bitwise identical to the Triton version."""
    x = x.float()
    a = x.abs().amax(dim=1)
    _, e = torch.frexp(a)
    e = torch.where(a > 0, e, torch.zeros_like(e)).to(torch.int32)
    sh = F - e
    scale = ((sh + 127) << 23).view(torch.float32)  # 2^sh built from bits, exact
    q = (x * scale[:, None]).to(torch.int32)  # power-of-two scaling is exact; float -> int truncates toward zero
    if F == 15:
        l = (q & 255) - 128
        h = (q - l - 128) >> 8
        cs = (256 * h + l).sum(1, dtype=torch.int32)
    else:
        l = ((q + 128) & 255) - 128
        h = (q - l) >> 8
        cs = torch.zeros_like(sh)
    return h.to(torch.int8), l.to(torch.int8), cs.to(torch.int32), sh.to(torch.int32)


def digitize_weight(w, F):
    return digitize_torch(w.float(), F)


if HAVE_TRITON:
    @triton.jit
    def _digitize_kernel(X, H, L, CS, SH, K, sx, sh_, F: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < K
        x = tl.load(X + row * sx + offs, mask=m, other=0.0).to(tl.float32)
        a = tl.max(tl.abs(x), axis=0)
        bits = a.to(tl.int32, bitcast=True)
        e = ((bits >> 23) & 0xFF) - 126
        e = tl.where(a > 0, e, 0)
        sh = F - e
        scale = ((sh + 127) << 23).to(tl.float32, bitcast=True)
        q = (x * scale).to(tl.int32)
        if F == 15:
            l = (q & 255) - 128
            h = (q - l - 128) >> 8
            cs = tl.sum(tl.where(m, 256 * h + l, 0), axis=0)
        else:
            l = ((q + 128) & 255) - 128
            h = (q - l) >> 8
            cs = sh * 0
        tl.store(H + row * sh_ + offs, h.to(tl.int8), mask=m)
        tl.store(L + row * sh_ + offs, l.to(tl.int8), mask=m)
        tl.store(CS + row, cs)
        tl.store(SH + row, sh)


def digitize(x, F, out=None):
    if not (HAVE_TRITON and x.is_cuda):
        return digitize_torch(x, F)
    R, K = x.shape
    if out is None:
        out = (torch.empty((R, K), dtype=torch.int8, device=x.device), torch.empty((R, K), dtype=torch.int8, device=x.device),
               torch.empty(R, dtype=torch.int32, device=x.device), torch.empty(R, dtype=torch.int32, device=x.device))
    h, l, cs, sh = out
    BLOCK = triton.next_power_of_2(K)
    _digitize_kernel[(R,)](x, h, l, cs, sh, K, x.stride(0), h.stride(0), F=F, BLOCK=BLOCK,
                           num_warps=16 if BLOCK >= 8192 else 8)
    return out


# ======================= exact linear layer =======================
def exact_linear_torch(xd, wd, F, trunc=False):
    xh, xl, cx, sx = [t.long() for t in xd]
    wh, wl, cw, sw = [t.long() for t in wd]
    K = xh.shape[1]
    tot = (xh @ wh.T) * 65536 + (xh @ wl.T + xl @ wh.T) * 256
    if not trunc:
        tot = tot + xl @ wl.T
    if F == 15:
        tot = tot + 128 * (cx[:, None] + cw[None, :]) + 128 * 128 * K
    e = -(sx[:, None] + sw[None, :])
    scale = ((e + 1023) << 52).view(torch.float64)
    return (tot.double() * scale).float()


if HAVE_TRITON:
    @triton.jit
    def _exact_kernel(XH, XL, WH, WL, CX, CW, SX, SW, OUT, WS, M, N, K, kconst, KCH, MN,
                      sxm, swn, som,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      TRUNC: tl.constexpr, OFFSET: tl.constexpr, GM: tl.constexpr, SK: tl.constexpr):
        """SK > 1 is split-K: program pid_k accumulates only K in [pid_k*KCH, (pid_k+1)*KCH) and writes its int64 partial
        sum to WS; _exact_fin then adds the partials in a fixed order. Integer addition is exact and order-independent,
        so the result is bitwise identical to SK = 1."""
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        width = GM * npn
        gid = pid // width
        first_m = gid * GM
        gsz = tl.minimum(npm - first_m, GM)
        pid_m = first_m + (pid % width) % gsz
        pid_n = (pid % width) // gsz
        rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)  # int64 so addresses do not overflow when M*N > 2^31
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        rk = tl.arange(0, BK)
        hh = tl.zeros((BM, BN), dtype=tl.int32)
        md = tl.zeros((BM, BN), dtype=tl.int32)
        ll = tl.zeros((BM, BN), dtype=tl.int32)
        k_lo = pid_k * KCH
        k_hi = tl.minimum(k_lo + KCH, K)
        for k0 in range(k_lo, k_hi, BK):
            kk = k0 + rk
            mx = (rm[:, None] < M) & (kk[None, :] < K)
            mw = (kk[:, None] < K) & (rn[None, :] < N)
            xh = tl.load(XH + rm[:, None] * sxm + kk[None, :], mask=mx, other=0)
            xl = tl.load(XL + rm[:, None] * sxm + kk[None, :], mask=mx, other=0)
            wh = tl.load(WH + rn[None, :] * swn + kk[:, None], mask=mw, other=0)
            wl = tl.load(WL + rn[None, :] * swn + kk[:, None], mask=mw, other=0)
            hh = tl.dot(xh, wh, hh, out_dtype=tl.int32)
            md = tl.dot(xh, wl, md, out_dtype=tl.int32)
            md = tl.dot(xl, wh, md, out_dtype=tl.int32)
            if not TRUNC:
                ll = tl.dot(xl, wl, ll, out_dtype=tl.int32)
        tot = (hh.to(tl.int64) << 16) + (md.to(tl.int64) << 8) + ll.to(tl.int64)
        if SK > 1:
            wsp = WS + pid_k.to(tl.int64) * MN + rm[:, None] * N + rn[None, :]
            tl.store(wsp, tot, mask=(rm[:, None] < M) & (rn[None, :] < N))
        else:
            if OFFSET:
                cx = tl.load(CX + rm, mask=rm < M, other=0).to(tl.int64)
                cw = tl.load(CW + rn, mask=rn < N, other=0).to(tl.int64)
                tot += 128 * (cx[:, None] + cw[None, :]) + kconst
            sx = tl.load(SX + rm, mask=rm < M, other=0)
            sw = tl.load(SW + rn, mask=rn < N, other=0)
            e = (-(sx[:, None] + sw[None, :])).to(tl.int64)
            scale = ((e + 1023) << 52).to(tl.float64, bitcast=True)
            out = (tot.to(tl.float64) * scale).to(tl.float32)  # fp64 exact -> fp32 one rounding; bf16 output rounds once more (same bits as fp32 output then cast)
            tl.store(OUT + rm[:, None] * som + rn[None, :], out.to(OUT.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def _exact_fin(WS, CX, CW, SX, SW, OUT, MN, N, kconst, som, SK: tl.constexpr, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        msk = idx < MN
        r = idx // N
        c = idx % N
        tot = tl.zeros((BLOCK,), dtype=tl.int64)
        for s_ in tl.static_range(SK):
            tot += tl.load(WS + s_ * MN + idx, mask=msk, other=0)
        if OFFSET:
            cx = tl.load(CX + r, mask=msk, other=0).to(tl.int64)
            cw = tl.load(CW + c, mask=msk, other=0).to(tl.int64)
            tot += 128 * (cx + cw) + kconst
        sx = tl.load(SX + r, mask=msk, other=0)
        sw = tl.load(SW + c, mask=msk, other=0)
        e = (-(sx + sw)).to(tl.int64)
        scale = ((e + 1023) << 52).to(tl.float64, bitcast=True)
        out = (tot.to(tl.float64) * scale).to(tl.float32)
        tl.store(OUT + r * som + c, out.to(OUT.dtype.element_ty), mask=msk)


if HAVE_TRITON:
    @triton.jit
    def _exact_bf16_kernel(XH, XL, WH, WL, CX, CW, SX, SW, OUT, WS, M, N, K, kconst, KCH, MN,
                           sxm, swn, som,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           TRUNC: tl.constexpr, OFFSET: tl.constexpr, GM: tl.constexpr, SK: tl.constexpr,
                           KMAX: tl.constexpr):
        """Same digits and same result bits as _exact_kernel, computed with bf16 tensor cores instead of INT8 ones
        (for GPUs with weak INT8 throughput, e.g. B300). INT8 digits (|d| <= 128) convert to bf16 exactly; each digit
        product is an integer of at most 2^14 (2^13 for the terms kept by F=14 truncation), and each fp32 accumulator
        only sums KMAX terms before being moved into an int32 accumulator, so every partial sum is an integer of
        magnitude <= 2^24 and is exact in fp32. With split-K chunks of <= KMAX the inner readout happens only once."""
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        width = GM * npn
        gid = pid // width
        first_m = gid * GM
        gsz = tl.minimum(npm - first_m, GM)
        pid_m = first_m + (pid % width) % gsz
        pid_n = (pid % width) // gsz
        rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        rk = tl.arange(0, BK)
        hh = tl.zeros((BM, BN), dtype=tl.int32)
        md = tl.zeros((BM, BN), dtype=tl.int32)
        ll = tl.zeros((BM, BN), dtype=tl.int32)
        k_lo = pid_k * KCH
        k_hi = tl.minimum(k_lo + KCH, K)
        for c0 in range(k_lo, k_hi, KMAX):
            c1 = tl.minimum(c0 + KMAX, k_hi)
            fh = tl.zeros((BM, BN), dtype=tl.float32)
            f1 = tl.zeros((BM, BN), dtype=tl.float32)
            f2 = tl.zeros((BM, BN), dtype=tl.float32)
            fl = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(c0, c1, BK):
                kk = k0 + rk
                mx = (rm[:, None] < M) & (kk[None, :] < K)
                mw = (kk[:, None] < K) & (rn[None, :] < N)
                xh = tl.load(XH + rm[:, None] * sxm + kk[None, :], mask=mx, other=0).to(tl.bfloat16)
                xl = tl.load(XL + rm[:, None] * sxm + kk[None, :], mask=mx, other=0).to(tl.bfloat16)
                wh = tl.load(WH + rn[None, :] * swn + kk[:, None], mask=mw, other=0).to(tl.bfloat16)
                wl = tl.load(WL + rn[None, :] * swn + kk[:, None], mask=mw, other=0).to(tl.bfloat16)
                fh = tl.dot(xh, wh, fh)
                f1 = tl.dot(xh, wl, f1)
                f2 = tl.dot(xl, wh, f2)
                if not TRUNC:
                    fl = tl.dot(xl, wl, fl)
            hh += fh.to(tl.int32)
            md += f1.to(tl.int32) + f2.to(tl.int32)
            if not TRUNC:
                ll += fl.to(tl.int32)
        tot = (hh.to(tl.int64) << 16) + (md.to(tl.int64) << 8) + ll.to(tl.int64)
        if SK > 1:
            wsp = WS + pid_k.to(tl.int64) * MN + rm[:, None] * N + rn[None, :]
            tl.store(wsp, tot, mask=(rm[:, None] < M) & (rn[None, :] < N))
        else:
            if OFFSET:
                cx = tl.load(CX + rm, mask=rm < M, other=0).to(tl.int64)
                cw = tl.load(CW + rn, mask=rn < N, other=0).to(tl.int64)
                tot += 128 * (cx[:, None] + cw[None, :]) + kconst
            sx = tl.load(SX + rm, mask=rm < M, other=0)
            sw = tl.load(SW + rn, mask=rn < N, other=0)
            e = (-(sx[:, None] + sw[None, :])).to(tl.int64)
            scale = ((e + 1023) << 52).to(tl.float64, bitcast=True)
            out = (tot.to(tl.float64) * scale).to(tl.float32)
            tl.store(OUT + rm[:, None] * som + rn[None, :], out.to(OUT.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


EXACT_CFGS = [(64, 128, 128, 4, 4, 8), (64, 64, 128, 4, 5, 8), (128, 128, 128, 8, 3, 8), (128, 64, 128, 4, 4, 8),
              (64, 256, 128, 8, 3, 8), (32, 128, 128, 4, 4, 8), (16, 128, 128, 4, 4, 8), (16, 256, 128, 8, 3, 8),
              (128, 128, 64, 4, 4, 8), (128, 256, 64, 8, 3, 8), (64, 128, 256, 4, 3, 8), (128, 128, 256, 8, 2, 8),
              (64, 256, 64, 4, 4, 8), (256, 128, 64, 8, 3, 8), (128, 64, 256, 4, 3, 8), (64, 64, 256, 4, 3, 8)]
_CFG_CACHE = {}


def exact_linear(xd, wd, F, trunc=False, out=None, cfg=None):
    if not (HAVE_TRITON and xd[0].is_cuda):
        return exact_linear_torch(xd, wd, F, trunc)
    xh, xl, cx, sx = xd
    wh, wl, cw, sw = wd
    M, K = xh.shape
    N = wh.shape[0]
    if out is None:
        out = torch.empty((M, N), device=xh.device, dtype=torch.float32)
    key = (M, N, K, F, trunc)
    cfg = cfg or _CFG_CACHE.get(key) or autotune_exact(xd, wd, F, trunc)
    BM, BN, BK, warps, stages, GM = cfg[:6]
    SK = cfg[6] if len(cfg) > 6 else 1
    BF16 = len(cfg) > 7 and cfg[7] == 1  # 1 = bf16 tensor cores (same result bits), 0 = INT8 tensor cores
    KCH = triton.cdiv(triton.cdiv(K, SK), BK) * BK
    SK = triton.cdiv(K, KCH)
    ws = torch.empty((SK, M, N), device=xh.device, dtype=torch.int64) if SK > 1 else out
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), SK)
    args = (xh, xl, wh, wl, cx, cw, sx, sw, out, ws, M, N, K, 128 * 128 * K, KCH, M * N, xh.stride(0), wh.stride(0), out.stride(0))
    if BF16:
        # largest digit product: 2^13 for the terms kept by F=14 truncation, 2^14 otherwise; sums must stay <= 2^24
        KMAX = 2048 if (F == 14 and trunc) else 1024
        assert KMAX % BK == 0
        _exact_bf16_kernel[grid](*args, BM=BM, BN=BN, BK=BK, TRUNC=trunc, OFFSET=(F == 15), GM=GM, SK=SK, KMAX=KMAX,
                                 num_warps=warps, num_stages=stages)
    else:
        _exact_kernel[grid](*args, BM=BM, BN=BN, BK=BK, TRUNC=trunc, OFFSET=(F == 15), GM=GM, SK=SK,
                            num_warps=warps, num_stages=stages)
    if SK > 1:
        BLOCK = 1024
        _exact_fin[(triton.cdiv(M * N, BLOCK),)](ws, cx, cw, sx, sw, out, M * N, N, 128 * 128 * K, out.stride(0),
                                                SK=SK, OFFSET=(F == 15), BLOCK=BLOCK, num_warps=4)
    return out


def autotune_exact(xd, wd, F, trunc):
    M, K = xd[0].shape
    N = wd[0].shape[0]
    key = (M, N, K, F, trunc)
    best = None
    for c in EXACT_CFGS:
        if c[0] > max(16, triton.next_power_of_2(M)):
            continue
        try:
            fn = lambda: exact_linear(xd, wd, F, trunc, cfg=c)
            fn()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(10):
                fn()
            en.record()
            torch.cuda.synchronize()
            t = st.elapsed_time(en)
            if best is None or t < best[0]:
                best = (t, c)
        except Exception:
            pass
    _CFG_CACHE[key] = best[1]
    return best[1]


