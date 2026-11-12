"""vllm-exact：让 vLLM 在 Qwen3.5 混合结构（GDN + 全注意力）上批组成无关，并尽量不变慢。

用法（必须在 import vllm 之前调用 enable）：

    import vllm_exact
    kw = vllm_exact.enable(max_model_len=16784)
    from vllm import LLM
    llm = LLM(model=..., dtype="bfloat16", max_model_len=16784, **kw)

enable() 做的事：
1. 打开 vLLM 的确定性模式（VLLM_BATCH_INVARIANT=1），并让引擎核心和本进程同进程（VLLM_ENABLE_V1_MULTIPROCESSING=0，
   插件和补丁才对引擎生效）；
2. 绕过 vLLM 对 GDN 后端的拦截（vLLM 0.32 的 GDN 后端声明不支持批组成无关，启动时报错；配合下面第 4 条的设置实测无关）；
3. 注册 quantization="exact_int8"：所有线性层（默认含 lm_head）换成 INT8 两数位精确累加；
4. 返回必须传给 LLM 的参数：不切 prompt（max_num_batched_tokens 足够大）、GDN prefill 用 Triton 后端、
   注意力后端 FLASH_ATTN（decode 步再换成本包的 bf16x2 kernel）、关前缀缓存；
5. 给 FlashAttentionImpl.forward 打补丁，decode 注意力换成本包的分页 bf16x2 kernel。
"""
import os
import sys
from pathlib import Path

__version__ = "0.1.0"
_ENABLED = False
TABLES = {"h200": Path(__file__).with_name("cfg_table_h200_F14.json")}


def enable(max_model_len, f=14, lm_head=True, attention=True, cfg_table="h200", max_num_batched_tokens=131072):
    """返回要传给 vllm.LLM(...) 的关键字参数。

    max_model_len：和传给 LLM 的一致（注意力 kernel 按它固定分段数）。
    f：14（截断成 3 次 INT8 dot，默认、最快）或 15（4 次 dot）。
    cfg_table："h200"（随包附带，H200 上用 CUDA graph 计时调的）、自己的表的路径，或 None（用内置的三档配置）。
      表只影响速度，不影响结果。表里只有 F=14 的配置；f=15 时建议传 None 或自己调。
    max_num_batched_tokens：必须大于任何一条 prompt 的长度（prompt 被切段时 GDN 的数值会随切法变）。
    """
    global _ENABLED
    if _ENABLED:
        raise RuntimeError("vllm_exact.enable() 只能调用一次")
    if "vllm" in sys.modules:
        raise RuntimeError("vllm_exact.enable() 必须在 import vllm 之前调用")
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from . import plugin  # 注册 exact_int8（这里第一次 import vllm）
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
