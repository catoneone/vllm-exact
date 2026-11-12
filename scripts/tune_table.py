"""为精确插件生成调参表：每个线性层形状 (N, K) × 每档批大小 M，从候选配置里选 GPU 时间最短的。

python scripts/tune_table.py OUT.json [--f 14] [--shapes 10240x4096,4096x4096,...]

- 计时把 20 次调用录进 CUDA graph 再回放，只算 GPU 时间（vLLM decode 用 CUDA graph；普通循环在小矩阵上测到的是
  Python 发射 kernel 的开销，会错误地偏向少发射的配置、选不上 split-K）。
- 每个候选配置的输出都和第一个配置比对，不逐位相同就报错（整数精确累加与分块无关，应当永远相同）。
- 默认形状是 Qwen3.5-9B（MiMo-V2.6-Distill-Qwen-9B）的：qkv、o/out_proj、GDN in_proj_qkvz、in_proj_ba、gate_up、down、lm_head。
"""
import argparse
import itertools
import json

import torch

from vllm_exact import kernels as K

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--f", type=int, default=14)
ap.add_argument("--shapes", default="10240x4096,4096x4096,12288x4096,64x4096,24576x4096,4096x12288,248320x4096")
args = ap.parse_args()
F, TR = args.f, args.f == 14
SHAPES = [tuple(map(int, s.split("x"))) for s in args.shapes.split(",")]
MS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512]
BASE = sorted(set([c[:6] for c in K.EXACT_CFGS] + [(bm, bn, bk, w, st, 8) for bm, bn, bk, w, st in itertools.product(
    (16, 32, 64), (64, 128, 256), (128, 256), (4, 8), (3, 4))]))
CFGS = [c + (sk,) for c in BASE for sk in (1, 2, 4, 8)]


def bench(fn, n=20):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / n * 1000


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    table, report = {}, []
    for (N, Kd) in SHAPES:
        w = torch.randn(N, Kd, device="cuda", generator=gen) * 0.02
        wd = K.digitize(w, F)
        for M in MS:
            x = torch.randn(M, Kd, device="cuda", generator=gen)
            xd = K.digitize(x, F)
            ref, best = None, None
            for c in CFGS:
                if c[0] > max(16, 1 << (M - 1).bit_length()):
                    continue
                if c[6] > 1 and (M > 128 or Kd // c[6] < c[2]):
                    continue
                try:
                    y = K.exact_linear(xd, wd, F, TR, cfg=c)
                except Exception:
                    continue  # 这个配置编译不了（共享内存不够等）
                if ref is None:
                    ref = y
                elif not torch.equal(y, ref):
                    raise RuntimeError(f"配置 {c} 改变了结果（N={N} K={Kd} M={M}）")
                t = bench(lambda c=c: K.exact_linear(xd, wd, F, TR, cfg=c))
                if best is None or t < best[0]:
                    best = (t, c)
            xb, wb = x.to(torch.bfloat16), w.to(torch.bfloat16)
            bt = bench(lambda: xb @ wb.T)
            table.setdefault(f"{N},{Kd}", {})[str(M)] = list(best[1])
            report.append(dict(N=N, K=Kd, M=M, exact_us=best[0], cublas_bf16_us=bt, cfg=best[1]))
            print(report[-1], flush=True)
    json.dump(table, open(args.out, "w"), indent=1)
    json.dump(report, open(args.out.replace(".json", "_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
