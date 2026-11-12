"""Batch-invariance check + decode speed.

python scripts/check_invariance.py MODEL_DIR [--mode exact|vllm-bi|base] [--texts T.jsonl.gz] [--speed] [--out OUT.json]

mode：
  exact    vllm_exact.enable() (default)
  vllm-bi  vLLM's own batch-invariant mode only (for comparison; vLLM 0.32 refuses to start on GDN models)
  base     plain vLLM
Check: 16 target sequences (1,024-token prompts, fixed seeds, 256 generated tokens each).
  A = the 16 alone; B = interleaved into a batch of 64; C = the 16 alone again.
  Compares generated tokens and per-token logprobs bitwise; once at temperature 1.0 (top_k 20, top_p 0.95) and once
  greedy.
--texts: jsonl.gz, one {"prompt_ids": [...], "out_ids": [...]} per line, concatenated and cut to length; without it,
  random tokens with a fixed seed.
--speed: steady-state decode step = (t(321) - t(65)) / 256, min of two runs each; scenarios (batch 256, 1K),
  (64, 8K), (16, 16K).
"""
import argparse
import gzip
import json
import os
import random
import time

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--mode", default="exact", choices=["exact", "vllm-bi", "base"])
ap.add_argument("--texts")
ap.add_argument("--speed", action="store_true")
ap.add_argument("--speed-only", action="store_true")
ap.add_argument("--out", default="invariance.json")
ap.add_argument("--max-model-len", type=int, default=16384 + 400)
args = ap.parse_args()

kw = {}
if args.mode == "exact":
    import vllm_exact
    kw = vllm_exact.enable(max_model_len=args.max_model_len)
elif args.mode == "vllm-bi":
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
from vllm import LLM, SamplingParams  # noqa: E402

_SEQS = None


def build_prompts(n, ctx):
    global _SEQS
    if _SEQS is None:
        if args.texts:
            items = [json.loads(l) for l in gzip.open(args.texts, "rt")]
            items.sort(key=lambda d: -len(d["out_ids"]))
            _SEQS = [d["prompt_ids"] + d["out_ids"] for d in items]
        else:
            rng = random.Random(0)
            _SEQS = [[rng.randrange(1000, 150000) for _ in range(4096)] for _ in range(64)]
    out, i = [], 0
    while len(out) < n:
        t = []
        while len(t) < ctx:
            t += _SEQS[i % len(_SEQS)]
            i += 1
        out.append(t[:ctx])
    return out


def toks_lps(o):
    c = o.outputs[0]
    lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)] if c.logprobs else []
    return list(c.token_ids), lps


def cmp(X, Y):
    first = [next((t for t in range(len(x[0])) if t >= len(y[0]) or x[0][t] != y[0][t]), None) for x, y in zip(X, Y)]
    return dict(seqs_same_tokens=sum(x[0] == y[0] for x, y in zip(X, Y)), seqs_same_logprobs=sum(x[1] == y[1] for x, y in zip(X, Y)),
                first_token_divergence=first,
                max_logprob_diff=max((abs(a - b) for x, y in zip(X, Y) for a, b in zip(x[1], y[1])), default=0.0))


def main():
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.88, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0}, **{"enable_prefix_caching": False, **kw})
    res = dict(mode=args.mode, invariance={})
    P = build_prompts(64, 1024)
    tgt, oth = P[:16], P[16:]
    order, tpos = [], []
    for j in range(16):  # interleave targets: t0, o0, o1, o2, t1, o3, ...
        tpos.append(len(order))
        order.append(tgt[j])
        order += oth[3 * j:3 * j + 3]
    for temp in (() if args.speed_only else (1.0, 0.0)):
        mk = lambda s: SamplingParams(temperature=temp, top_k=20 if temp > 0 else -1, top_p=0.95 if temp > 0 else 1.0,
                                      max_tokens=256, seed=s, logprobs=0, ignore_eos=True)
        spA = [mk(1000 + j) for j in range(16)]
        A = [toks_lps(o) for o in llm.generate([dict(prompt_token_ids=p) for p in tgt], spA, use_tqdm=False)]
        spB = [mk(1000 + tpos.index(i)) if i in tpos else mk(1100 + i) for i in range(len(order))]
        outB = llm.generate([dict(prompt_token_ids=p) for p in order], spB, use_tqdm=False)
        B = [toks_lps(outB[i]) for i in tpos]
        C = [toks_lps(o) for o in llm.generate([dict(prompt_token_ids=p) for p in tgt], spA, use_tqdm=False)]
        r = dict(A_vs_B=cmp(A, B), A_vs_C=cmp(A, C))
        res["invariance"][f"T{temp}"] = r
        print("invariance", temp, json.dumps({k: (v["seqs_same_tokens"], v["seqs_same_logprobs"]) for k, v in r.items()}), flush=True)
    if args.speed or args.speed_only:
        res["speed"] = []
        for (B, ctx) in ((256, 1024), (64, 8192), (16, 16384)):
            prompts = [dict(prompt_token_ids=p) for p in build_prompts(B, ctx)]

            def gen(n):
                sp = [SamplingParams(temperature=1.0, top_k=20, top_p=0.95, max_tokens=n, ignore_eos=True, seed=j) for j in range(B)]
                t = time.time()
                llm.generate(prompts, sp, use_tqdm=False)
                return time.time() - t
            gen(321)
            step = (min(gen(321), gen(321)) - min(gen(65), gen(65))) / 256
            res["speed"].append(dict(B=B, ctx=ctx, step_ms=step * 1000, tok_per_s=B / step))
            print("speed", json.dumps(res["speed"][-1]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
