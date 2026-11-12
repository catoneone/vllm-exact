"""GPU 单元测试（需要 CUDA + Triton）：python -m pytest tests/test_gpu.py
不需要 vLLM：只测 kernels 和 paged_attn 的 kernel 本身。
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
DEV = "cuda"


@pytest.mark.parametrize("F,tr", [(14, True), (15, False)])
@pytest.mark.parametrize("M,N,Kd", [(24, 384, 4096), (5, 200, 12288), (1, 64, 4096)])
def test_exact_linear_split_k_bitwise(F, tr, M, N, Kd):
    """任何分块配置、任何 split-K 拆法都和 CPU int64 参考逐位相同；直接写 bf16 和先出 fp32 再转逐位相同。"""
    from vllm_exact import kernels as K
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(M, Kd, device=DEV, generator=g)
    x[:, 7] *= 300
    w = torch.randn(N, Kd, device=DEV, generator=g) * 0.02
    xd, wd = K.digitize(x, F), K.digitize(w, F)
    ref = K.exact_linear_torch([t.cpu() for t in xd], [t.cpu() for t in wd], F, tr)
    for sk in (1, 2, 4, 8):
        for cfg in ((16, 64, 128, 4, 3, 8, sk), (64, 128, 128, 4, 4, 8, sk), (16, 128, 256, 8, 3, 8, sk)):
            y = K.exact_linear(xd, wd, F, tr, cfg=cfg)
            assert torch.equal(y.cpu(), ref), cfg
            yb = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
            K.exact_linear(xd, wd, F, tr, out=yb, cfg=cfg)
            assert torch.equal(yb, ref.to(DEV).to(torch.bfloat16)), cfg


@pytest.mark.parametrize("F", [14, 15])
def test_digitize_bf16_input(F):
    from vllm_exact import kernels as K
    g = torch.Generator(device=DEV).manual_seed(1)
    xb = (torch.randn(300, 4096, device=DEV, generator=g) * 3).to(torch.bfloat16)
    for a, b in zip(K.digitize(xb, F), K.digitize(xb.float().contiguous(), F)):
        assert torch.equal(a, b)
    for a, b in zip(K.digitize(xb.float().contiguous(), F), K.digitize_torch(xb.float().cpu(), F)):
        assert torch.equal(a.cpu(), b)


def _make(B, L, BS, H=16, KV=4, D=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nbp = (L + BS - 1) // BS
    nblocks = B * nbp + 7
    kv = (torch.randn(nblocks, KV, BS, 2 * D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    bt = torch.randperm(nblocks, device=DEV, generator=g)[:B * nbp].view(B, nbp).to(torch.int32)
    q = torch.randn(B, H, D, device=DEV, generator=g).to(torch.bfloat16)
    sl = torch.randint(L // 2, L + 1, (B,), device=DEV, generator=g).to(torch.int32)
    return q, kv, bt, sl


@pytest.mark.parametrize("B,L,BS", [(8, 300, 16), (8, 5000, 528), (4, 9000, 528), (64, 8192, 528)])
def test_paged_attention(B, L, BS):
    """误差 vs fp32 参考；只算前 n 个请求时前 n 个逐位不变；物理页搬家（块表跟着改）逐位不变。"""
    from vllm_exact import paged_attn as P
    q, kv, bt, sl = _make(B, L, BS)
    kc, vc = kv.transpose(1, 2).split(256, dim=-1)
    o = torch.empty(B, 16, 256, device=DEV, dtype=torch.bfloat16)
    P.paged_decode(q, kc, vc, bt, sl, 1 / 16, o, B, int(sl.max()))
    n = min(B, 4)
    ref = P.paged_decode_torch(q, kc, vc, bt, sl, 1 / 16, n)
    assert float((o[:n].float() - ref).abs().max()) < 1e-3
    o2 = torch.empty_like(o)
    P.paged_decode(q, kc, vc, bt, sl, 1 / 16, o2, n, int(sl.max()))
    assert torch.equal(o[:n], o2[:n])
    newpos = torch.randperm(kv.shape[0], device=DEV)
    kv2 = torch.empty_like(kv)
    kv2[newpos] = kv
    kc2, vc2 = kv2.transpose(1, 2).split(256, dim=-1)
    o3 = torch.empty_like(o)
    P.paged_decode(q, kc2, vc2, newpos[bt.long()].to(torch.int32), sl, 1 / 16, o3, B, int(sl.max()))
    assert torch.equal(o, o3)
