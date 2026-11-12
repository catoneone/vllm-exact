"""vllm-exact: batch-invariant vLLM inference for the Qwen3.5 hybrid architecture (GDN + full attention), with as
little slowdown as possible.

Usage (enable() must be called before importing vllm):

    import vllm_exact
    kw = vllm_exact.enable(max_model_len=16784)
    from vllm import LLM
    llm = LLM(model=..., dtype="bfloat16", max_model_len=16784, **kw)

What enable() does:
1. turns on vLLM's batch-invariant mode (VLLM_BATCH_INVARIANT=1) and runs the engine core in this process
   (VLLM_ENABLE_V1_MULTIPROCESSING=0, so the plugin and patches apply to the engine);
2. bypasses vLLM's check on the GDN backend (vLLM 0.32's GDN backend declares no batch-invariance support and refuses
   to start; with the settings in step 4 it is batch-invariant in our tests);
3. registers quantization="exact_int8": every linear layer (lm_head included by default) uses exact INT8 two-digit
   accumulation;
4. returns the arguments that must be passed to LLM: prompts are never chunked (large max_num_batched_tokens), GDN
   prefill uses the Triton backend, attention backend FLASH_ATTN (decode steps then use this package's bf16x2 kernel),
   prefix caching off;
5. patches FlashAttentionImpl.forward so decode attention uses this package's paged bf16x2 kernel.
"""
import os
import sys
from pathlib import Path

__version__ = "0.1.0"
_ENABLED = False
TABLES = {"h200": Path(__file__).with_name("cfg_table_h200_F14.json")}


def enable(max_model_len, f=14, lm_head=True, attention=True, cfg_table="h200", max_num_batched_tokens=131072):
    """Returns keyword arguments to pass to vllm.LLM(...).

    max_model_len: same value as passed to LLM (the attention kernel fixes its number of segments from it).
    f: 14 (truncated, 3 INT8 dots; default and fastest) or 15 (4 dots).
    cfg_table: "h200" (shipped, tuned on H200 with CUDA-graph timing), a path to your own table, or None (three built-in
      configs). The table only affects speed, never results. It only has F=14 entries; with f=15 pass None or tune.
    max_num_batched_tokens: must exceed the length of every prompt (when a prompt is chunked, GDN numerics depend on
      where the chunks fall).
    """
    global _ENABLED
    if _ENABLED:
        raise RuntimeError("vllm_exact.enable() can only be called once")
    if "vllm" in sys.modules:
        raise RuntimeError("vllm_exact.enable() must be called before importing vllm")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from . import plugin  # registers exact_int8 (first import of vllm happens here)
    plugin.F = f
    plugin.TRUNC = f == 14
    plugin.LMHEAD = lm_head
    plugin.load_table(TABLES.get(cfg_table, cfg_table) if cfg_table else None)

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
    GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)

    kw = dict(quantization="exact_int8", max_num_batched_tokens=max_num_batched_tokens, enable_prefix_caching=False,
              additional_config={"gdn_prefill_backend": "triton"})
    if attention:
        from . import paged_attn
        paged_attn.MAX_LEN = max_model_len
        paged_attn.install()
        kw["attention_config"] = {"backend": "FLASH_ATTN"}
    _ENABLED = True
    return kw
