"""分页 KV 缓存上的 bf16x2 decode 注意力 kernel，以及给 vLLM FlashAttention 后端打的补丁。

kernel：每个（请求, KV 头, SPLIT 个 key 一段）一个程序；段长固定、段数按模型最大长度固定（与批无关，CUDA graph 回放时网格不变），
  段间按固定顺序合并。
  QKᵀ：q、K 都是 bf16 → 一次 bf16 dot，乘积在 fp32 里精确；softmax 权重 p 拆成高低两半 → 两次 bf16 dot × V（bf16，精确）。
  KV 布局沿用 vLLM FlashAttention：kv_cache [num_blocks, num_kv_heads, block_size, 2*head_size]，
  key_cache = kv_cache.transpose(1, 2)[..., :D]，value_cache = [..., D:]（视图，按 stride 访问）。
补丁：FlashAttentionImpl.forward
  - 纯 decode 步（max_query_len == 1）：全部走本 kernel（CUDA graph 安全，无 CPU 同步）；
  - 混合步（有 prefill）：先走 vLLM 原生实现，再把开头的 decode token（vLLM 对混合模型会把 decode 排在前面）用本 kernel 覆盖。
  每个 decode token 永远由本 kernel 算，prefill token 永远由原生实现算，所以与批组成无关。
"""
import torch
import triton
import triton.language as tl

# kernel 参数是全局常数（不随批变化）。BLK / SPLIT 改变浮点运算顺序，所以一次部署只能用一组；
# 默认值是在 H200、页大小 528（vLLM 给 Qwen3.5 混合结构选的）上调出来的。由 vllm_exact.enable() 设置。
SPLIT = 4096
BLK = 64
WARPS = 4
STAGES = 4
MAX_LEN = 16384  # 段数按模型最大长度固定；enable() 里设成 max_model_len


@triton.jit
def _paged_partial(Q, KC, VC, BT, SL, MP, LP, AP, scale,
                   sqt, sqh, skb, skp, skh, sbt, NS,
                   G: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr, SPL: tl.constexpr, BS: tl.constexpr):
    b = tl.program_id(0)
    g = tl.program_id(1)
    s_id = tl.program_id(2)
    L = tl.load(SL + b)
    rq = tl.arange(0, 16)
    dq = tl.arange(0, D)
    qm = rq < G
    q = tl.load(Q + b * sqt + (g * G + rq)[:, None] * sqh + dq[None, :], mask=qm[:, None], other=0.0).to(tl.bfloat16)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, D), tl.float32)
    start = s_id * SPL
    end = tl.minimum(start + SPL, L)
    for k0 in range(start, end, BLK):
        kk = k0 + tl.arange(0, BLK)
        km = kk < end
        page = tl.load(BT + b * sbt + kk // BS, mask=km, other=0).to(tl.int64)
        off = (kk % BS).to(tl.int64)
        base = page * skb + off * skp + g * skh
        k = tl.load(KC + base[:, None] + dq[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(km[None, :], s, float("-inf"))
        mb = tl.max(s, axis=1)
        mn = tl.maximum(m, mb)
        alpha = tl.exp(m - mn)
        p = tl.exp(s - mn[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(VC + base[:, None] + dq[None, :], mask=km[:, None], other=0.0)
        p_hi = p.to(tl.bfloat16)
        p_lo = (p - p_hi.to(tl.float32)).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p_hi, v) + tl.dot(p_lo, v)
        m = mn
    hrow = g * G + rq
    H = G * tl.num_programs(1)
    idx = (b * H + hrow).to(tl.int64) * NS + s_id
    tl.store(MP + idx, m, mask=qm)
    tl.store(LP + idx, l, mask=qm)
    tl.store(AP + idx[:, None] * D + dq[None, :], acc, mask=qm[:, None])


@triton.jit
def _paged_merge(MP, LP, AP, O, sot, soh, H, NS, D: tl.constexpr):
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    dq = tl.arange(0, D)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros((D,), tl.float32)
    for s in range(0, NS):
        ms = tl.load(MP + bh * NS + s)
        ls = tl.load(LP + bh * NS + s)
        a_s = tl.load(AP + (bh.to(tl.int64) * NS + s) * D + dq)
        mn = tl.maximum(m, ms)
        c_old = tl.where(m == float("-inf"), 0.0, tl.exp(m - mn))
        c_new = tl.where(ms == float("-inf"), 0.0, tl.exp(ms - mn))
        l = l * c_old + ls * c_new
        acc = acc * c_old + a_s * c_new
        m = mn
    tl.store(O + b * sot + h * soh + dq, (acc / l).to(O.dtype.element_ty))


def paged_decode(query, key_cache, value_cache, block_table, seq_lens, scale, out, n, max_seq_len, blk=None, warps=None,
                 stages=None, split=None):
    """query: (≥n, H, D)；key/value_cache: (num_blocks, block_size, KV, D) 视图；out: (≥n, H, D)。只算前 n 个请求。"""
    H, D = query.shape[1], query.shape[2]
    KV = key_cache.shape[2]
    BS = key_cache.shape[1]
    G = H // KV
    blk, warps, stages, split = blk or BLK, warps or WARPS, stages or STAGES, split or SPLIT
    NS = max(1, triton.cdiv(MAX_LEN, split))  # 超过序列长度的段直接空转，合并时贡献精确为 0
    mp = torch.empty(n * H * NS, device=query.device, dtype=torch.float32)
    lp = torch.empty_like(mp)
    ap = torch.empty(n * H * NS * D, device=query.device, dtype=torch.float32)
    _paged_partial[(n, KV, NS)](query, key_cache, value_cache, block_table, seq_lens, mp, lp, ap, scale,
                                query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1),
                                key_cache.stride(2), block_table.stride(0), NS,
                                G=G, D=D, BLK=blk, SPL=split, BS=BS, num_warps=warps, num_stages=stages)
    _paged_merge[(n * H,)](mp, lp, ap, out, out.stride(0), out.stride(1), H, NS, D=D, num_warps=4)
    return out


def paged_decode_torch(query, key_cache, value_cache, block_table, seq_lens, scale, n):
    BS = key_cache.shape[1]
    H = query.shape[1]
    KV = key_cache.shape[2]
    G = H // KV
    out = torch.empty(n, H, query.shape[2], device=query.device, dtype=torch.float32)
    for b in range(n):
        L = int(seq_lens[b])
        pos = torch.arange(L, device=query.device)
        pages = block_table[b, pos // BS].long()
        k = key_cache[pages, pos % BS].float()  # (L, KV, D)
        v = value_cache[pages, pos % BS].float()
        for h in range(H):
            sc = (k[:, h // G] @ query[b, h].float()) * scale
            out[b, h] = torch.softmax(sc, 0) @ v[:, h // G]
    return out


def install():
    """给 vLLM 的 FlashAttentionImpl.forward 打补丁（需在 LLM 初始化前调用，且 VLLM_ENABLE_V1_MULTIPROCESSING=0）。"""
    from vllm.v1.attention.backends import flash_attn as fa
    orig = fa.FlashAttentionImpl.forward
    DEC = fa.AttentionType.DECODER

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        if attn_metadata is None or self.attn_type != DEC:
            return orig(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        D = self.head_size
        key_cache, value_cache = kv_cache.transpose(1, 2).split(D, dim=-1)
        out3 = output.view(output.shape[0], self.num_heads, D)
        if attn_metadata.max_query_len == 1:
            n = attn_metadata.num_actual_tokens
            if n > 0:
                paged_decode(query, key_cache, value_cache, attn_metadata.block_table, attn_metadata.seq_lens,
                             self.scale, out3, n, attn_metadata.max_seq_len)
            return output
        res = orig(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)
        qsl = attn_metadata.query_start_loc
        ql = (qsl[1:] - qsl[:-1])
        nd = int(torch.cumprod((ql == 1).to(torch.int32), 0).sum().item())  # 开头连续的 decode 请求数（混合步在 eager 下跑）
        if nd > 0:
            paged_decode(query, key_cache, value_cache, attn_metadata.block_table, attn_metadata.seq_lens,
                         self.scale, out3, nd, attn_metadata.max_seq_len)
        return res

    fa.FlashAttentionImpl.forward = forward
    return orig
