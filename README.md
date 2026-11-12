# vllm-exact

Makes vLLM **batch-invariant** on the Qwen3.5 hybrid architecture (Gated DeltaNet + full attention, e.g. MiMo-V2.6-Distill-Qwen-9B). A request gets bitwise-identical generated tokens and per-token logprobs whether it runs alone, alongside other requests, or again later. The slowdown is kept as small as possible.

Use it when inference results have to be reproducible as evidence, for example when other parties run evaluations and you spot-check their results by recomputing them.

> Tested only on **vLLM 0.32.0 / torch 2.13.0 / triton 3.7.1 / a single H200**. The patches rely on vLLM internals; re-verify before using other versions.

## Results (H200, MiMo-V2.6-Distill-Qwen-9B, bf16)

Batch-invariance check: 16 target sequences (1,024-token prompts, 256 generated tokens each) run alone, then interleaved into a batch of 64, then alone again; once at temperature 1.0 (top_k 20, top_p 0.95) and once greedy.

| Configuration | Sequences with identical tokens (T=1.0 / greedy, out of 16) | Identical logprobs |
|---|---|---|
| Plain vLLM | 6 / 10 | 0 / 0 |
| vLLM's batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) | refuses to start: GDN backend unsupported | |
| Batch-invariant mode + GDN check bypassed only | 5 / 6 | 0 / 0 |
| **vllm-exact** | **16 / 16** | **16 / 16** |

Reruns are bitwise identical as well.

Decode step time (ms, same machine):

| | batch 256, 1K context | batch 64, 8K | batch 16, 16K |
|---|---|---|---|
| Plain vLLM (not batch-invariant) | 18.8 | 12.0 | 8.1 |
| vLLM batch-invariant mode + the settings that make it work (without this package's kernels) | 23.2 (+23%) | 21.2 (+77%) | 21.8 (+169%) |
| **vllm-exact** | **24.6 (+31%)** | **13.0 (+9%)** | **8.5 (+5%)** |

Re-verified on another H200 after installing as described below: all 16 tests in `tests/` pass; `scripts/check_invariance.py` (random-token prompts) gives 16 / 16 bitwise identical; speed 24.9 / 12.9 / 8.6 ms, matching the table.

The second row is vLLM's batch-invariant mode plus the settings in item 1 below (also 16 / 16 bitwise identical), measured on another H200 of the same type.

## Usage

```python
import vllm_exact
kw = vllm_exact.enable(max_model_len=16784)   # must run before importing vllm

from vllm import LLM, SamplingParams
llm = LLM(model="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", dtype="bfloat16", max_model_len=16784, **kw)
```

Requirements:
- give every request a fixed `seed` (`SamplingParams(seed=...)`);
- every prompt must be shorter than `max_num_batched_tokens` (default 131072) so it is never chunked;
- no prefix caching (the arguments returned by `enable()` turn it off).

Install: `pip install -e .` (pulls in vllm==0.32.0).

## How it works

Batch dependence comes from floating-point addition not being associative: when the batch size changes, kernels tile the same dot product differently and add in a different order, the last bit changes, and long generations diverge. This package handles it in four parts:

1. **vLLM settings** (`enable()`): turn on vLLM's batch-invariant mode; bypass its check on the GDN backend; never chunk prompts (where a prompt is cut changes GDN's chunking); use the Triton backend for GDN prefill; turn off prefix caching. Not chunking and the Triton backend are both needed together; their individual effects were not separated. These settings alone already make vLLM batch-invariant, but 2.7× slower at small batch and long context, because vLLM's batch-invariant matmul is slow.
2. **Exact INT8-accumulation linear layers** (`plugin.py`, `kernels.py`, `quantization="exact_int8"`): for every linear layer and lm_head, activations and weights are converted per row to 14-bit fixed point and split into two INT8 digits, multiplied in INT8 tensor cores and accumulated in integers. Integer addition is exact and order-independent, so the result does not depend on tiling, batch size or split-K partitioning. The output path is int64 → fp64 (exact) → fp32 → bf16. The 14-bit truncated variant drops the low × low term and needs only 3 INT8 dots. The two digits of a weight take 2 bytes, so weight traffic equals bf16.
   - Split-K: integer partial sums can be added in any partition without changing a bit, so small matrices can split K across programs for more parallelism. Floating point cannot do this.
   - Tuning table (`cfg_table_h200_F14.json`): the fastest kernel config for each (output dim, input dim, batch size). It affects speed only, never results.
3. **bf16x2 decode attention** (`paged_attn.py`): reads vLLM's paged KV cache directly. QKᵀ is one bf16 dot (q and K are bf16; the products are exact in fp32). The softmax weights are split into high and low bf16 halves for two bf16 dots with V. Each request is split into fixed-length segments merged in a fixed order, independent of the batch. Reaches 4.0–4.1 TB/s on H200, close to the bandwidth limit.
4. **Patch**: `FlashAttentionImpl.forward` is patched so decode tokens always go through the kernel in item 3 and prefill tokens always go through vLLM's native implementation (in batch-invariant mode).

## Tests

```bash
python -m pytest tests/test_reference_cpu.py     # CPU: self-checks of the reference implementation
python -m pytest tests/test_gpu.py               # GPU: every split-K partition bitwise equal to the CPU integer reference; attention error and bitwise invariance
python scripts/check_invariance.py MODEL_DIR --speed   # end to end: batch invariance + decode speed
python scripts/tune_table.py my_table.json       # rebuild the tuning table for another GPU or model
```

`check_invariance.py --mode base` / `--mode vllm-bi` compare against plain vLLM and vLLM's own batch-invariant mode. Without `--texts`, prompts are random tokens with a fixed seed.

Tune with CUDA-graph timing (the script does this): a plain timing loop on small matrices measures Python kernel-launch overhead and never picks split-K.

## Not covered

- **Cross-GPU consistency**: the exact linear layers are hardware-independent, but vLLM's GDN, normalization, prefill attention and sampling are still floating point. Bitwise identity is only guaranteed on the same GPU type with the same software versions.
- **Accuracy**: KL against an fp32 reference has not been measured inside vLLM. In a separate deterministic engine (15-bit, untruncated) the mean KL was 4.1e-4, versus 1.5e-3 for bf16 inference. The accuracy of the 14-bit truncated variant has not been measured separately.
- **Prefill speed**: only decode was measured.
- **Multi-GPU / tensor parallelism** and other architectures: not tested.
- **Not chunking prompts** raises memory pressure when many long prompts prefill at once. A better fix would be for GDN to chunk at fixed absolute positions, which needs changes to vLLM's GDN kernels.
- About 30% overhead remains at large batch: at batch 256 the matmuls are close to compute-bound. Three INT8 dots cost 1.5× a bf16 dot in theory; this Triton kernel measures 1.9×.

## Files

```
src/vllm_exact/__init__.py         enable(): settings, GDN check bypass, plugin registration, attention patch
src/vllm_exact/plugin.py           quantization="exact_int8" (linear layers + lm_head)
src/vllm_exact/kernels.py          digitization and exact matmul (with split-K): Triton kernels and torch references
src/vllm_exact/paged_attn.py       paged bf16x2 decode attention + FlashAttentionImpl patch
src/vllm_exact/cfg_table_h200_F14.json   H200 tuning table
scripts/check_invariance.py        batch invariance + speed
scripts/tune_table.py              build a tuning table
tests/                             CPU / GPU unit tests
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
