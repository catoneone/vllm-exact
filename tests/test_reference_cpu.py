"""CPU 上能跑的参考实现自检（不需要 GPU、Triton、vLLM）：python -m pytest tests/test_reference_cpu.py"""
import torch

from vllm_exact import kernels as K


def _xw(M=7, N=33, Kd=512, seed=0, outlier=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, Kd, generator=g)
    if outlier:
        x[:, 3] *= 500  # 每行一个大值，检验按行缩放
    w = torch.randn(N, Kd, generator=g) * 0.02
    return x, w


def test_digitize_reconstructs():
    x, _ = _xw()
    for F in (14, 15):
        h, l, cs, sh = K.digitize_torch(x, F)
        q = 256 * h.long() + l.long() + (128 if F == 15 else 0)
        scale = torch.pow(2.0, -sh.double())[:, None]
        err = (q.double() * scale - x.double()).abs().max(1).values / x.abs().amax(1).double()
        assert float(err.max()) < 2.0 ** -(F - 1)


def test_exact_linear_close_to_float():
    """没有离群值时误差在 1e-3 以内。有离群值时，按行缩放会让同一行的小值变粗（定点方案的固有代价）。"""
    x, w = _xw(outlier=False)
    ref = x.double() @ w.double().T
    for F, tr in ((15, False), (14, True)):
        y = K.exact_linear(K.digitize_torch(x, F), K.digitize_torch(w, F), F, tr).double()
        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        assert rel < 1e-3, (F, tr, rel)


def test_rows_independent():
    """每行的结果只取决于这一行：单独算和放进批里算逐位相同（批组成无关的根本原因）。"""
    x, w = _xw(M=9)
    for F, tr in ((15, False), (14, True)):
        wd = K.digitize_torch(w, F)
        full = K.exact_linear(K.digitize_torch(x, F), wd, F, tr)
        for i in (0, 4, 8):
            one = K.exact_linear(K.digitize_torch(x[i:i + 1], F), wd, F, tr)
            assert torch.equal(one[0], full[i])


def test_k_split_is_exact():
    """把 K 拆成几段分别做整数累加再相加，和不拆逐位相同（split-K 的依据）。"""
    x, w = _xw(Kd=1024)
    F = 14
    xd, wd = K.digitize_torch(x, F), K.digitize_torch(w, F)
    xh, xl = xd[0].long(), xd[1].long()
    wh, wl = wd[0].long(), wd[1].long()
    whole = (xh @ wh.T) * 65536 + (xh @ wl.T + xl @ wh.T) * 256
    parts = sum((xh[:, s] @ wh[:, s].T) * 65536 + (xh[:, s] @ wl[:, s].T + xl[:, s] @ wh[:, s].T) * 256
                for s in (slice(0, 300), slice(300, 777), slice(777, 1024)))
    assert torch.equal(whole, parts)
