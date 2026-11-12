# vllm-exact

让 vLLM 在 Qwen3.5 混合结构（Gated DeltaNet + 全注意力，例如 MiMo-V2.6-Distill-Qwen-9B）上**批组成无关**：同一条请求，不管单独跑、和别的请求一起跑、还是重跑，生成的 token 和每个 token 的 logprob 都逐位相同。速度损失尽量小。

适用场景：需要把推理结果当成可复现"证据"的地方，比如让别人跑评测、自己抽查复算。

> 只在 **vLLM 0.32.0 / torch 2.13.0 / triton 3.7.1 / 单卡 H200** 上测过。补丁依赖 vLLM 内部接口，换版本必须重新验证。

## 结果（H200，MiMo-V2.6-Distill-Qwen-9B，bf16）

批组成无关测试：16 条目标序列（prompt 各 1,024 token，各生成 256 token）先单独跑，再穿插进 64 条的批里跑，再单独重跑；温度 1.0（top_k 20、top_p 0.95）和贪心各一次。

| 配置 | token 逐位相同（温度 1.0 / 贪心） | logprob 逐位相同 |
|---|---|---|
| 正常 vLLM | 6 / 10（共 16 条） | 0 / 0 |
| vLLM 自带确定性模式（`VLLM_BATCH_INVARIANT=1`） | 启动时报错：GDN 后端不支持 | |
| 确定性模式 + 只绕过 GDN 检查 | 5 / 6 | 0 / 0 |
| **vllm-exact** | **16 / 16** | **16 / 16** |

重跑也都逐位相同。

decode 每步耗时（毫秒，同一台机器）：

| | 批 256、上下文 1K | 批 64、8K | 批 16、16K |
|---|---|---|---|
| 正常 vLLM（不确定） | 18.8 | 12.0 | 8.1 |
| vLLM 确定性模式 + 让它能跑的设置（不用本包的 kernel） | 23.2（+23%） | 21.2（+77%） | 21.8（+169%） |
| **vllm-exact** | **24.6（+31%）** | **13.0（+9%）** | **8.5（+5%）** |

按本 README 的方式安装后，在另一台 H200 上复验：`tests/` 16 项全过；`scripts/check_invariance.py`（随机 token prompt）16 / 16 逐位相同；速度 24.9 / 12.9 / 8.6 ms，和上表一致。

第二行是 vLLM 自带确定性模式加上下面第 1 条的设置（同样 16 / 16 逐位相同），在另一台同型号 H200 上测的。

## 用法

```python
import vllm_exact
kw = vllm_exact.enable(max_model_len=16784)   # 必须在 import vllm 之前

from vllm import LLM, SamplingParams
llm = LLM(model="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", dtype="bfloat16", max_model_len=16784, **kw)
```

要求：
- 采样时给每条请求固定 `seed`（`SamplingParams(seed=...)`）；
- 每条 prompt 都要短于 `max_num_batched_tokens`（默认 131072），不能被切段；
- 不开前缀缓存（`enable()` 返回的参数里已经关了）。

安装：`pip install -e .`（会装 vllm==0.32.0）。

## 做了什么

批组成相关的根源是浮点加法不满足结合律：同一个点积，批大小不同时 kernel 的分块和累加顺序不同，结果差最后一位，生成长了就分叉。本包分四块处理：

1. **vLLM 设置**（`enable()`）：打开 vLLM 的确定性模式；绕过它对 GDN 后端的拦截；prompt 不切段（切段位置会改变 GDN 的分块）；GDN prefill 用 Triton 后端；关前缀缓存。不切段和 Triton 后端一起用才批组成无关，两者各自的作用没有拆开测。只做这一步就已经批组成无关，但小批量长上下文时慢 2.7 倍，因为 vLLM 的确定性矩阵乘很慢。
2. **INT8 精确累加线性层**（`plugin.py`、`kernels.py`，`quantization="exact_int8"`）：所有线性层和 lm_head 的激活、权重都按行做 14 位定点，拆成两个 INT8 数位，在 INT8 张量核里相乘，用整数累加。整数加法精确、和顺序无关，所以结果与分块、批大小、split-K 拆法都无关。最后 int64 → fp64（精确）→ fp32 → bf16。14 位截断丢掉低位 × 低位那一项，只做 3 次 INT8 dot。每个权重的两个数位共 2 字节，读权重的量和 bf16 一样。
   - split-K：整数部分和可以按任意拆法相加，结果逐位不变，所以小矩阵可以把 K 拆给多个程序来提高并行度。浮点做不到这一点。
   - 调参表（`cfg_table_h200_F14.json`）：每个（输出维、输入维、批大小）选最快的 kernel 配置。只影响速度，不影响结果。
3. **bf16x2 decode 注意力**（`paged_attn.py`）：直接读 vLLM 的分页 KV 缓存。QKᵀ 用一次 bf16 dot（q、K 都是 bf16，乘积在 fp32 里精确）；softmax 权重拆成高低两半，做两次 bf16 dot × V。每条请求按固定长度分段、按固定顺序合并，与批无关。H200 上已经到 4.0–4.1 TB/s，接近带宽上限。
4. **补丁**：给 `FlashAttentionImpl.forward` 打补丁，decode token 都走第 3 条的 kernel，prefill token 都走 vLLM 原生实现（确定性模式下）。

## 测试

```bash
python -m pytest tests/test_reference_cpu.py     # CPU：参考实现自检
python -m pytest tests/test_gpu.py               # GPU：split-K 所有拆法与 CPU 整数参考逐位相同、注意力误差和逐位不变性
python scripts/check_invariance.py MODEL_DIR --speed   # 端到端：批组成无关 + decode 测速
python scripts/tune_table.py my_table.json       # 换显卡或换模型时重新生成调参表
```

`check_invariance.py --mode base` / `--mode vllm-bi` 可以对照正常 vLLM 和 vLLM 自带的确定性模式。不给 `--texts` 时用固定种子的随机 token 作 prompt。

调参要用 CUDA graph 计时（脚本里已经这样做了）：普通循环在小矩阵上测到的是 Python 发射 kernel 的开销，会选不上 split-K。

## 没有覆盖的

- **跨显卡一致**：精确线性层与硬件无关，但 vLLM 的 GDN、归一化、prefill 注意力、采样仍是浮点，只保证同一种卡、同一套软件版本下逐位一致。
- **精度**：没有在 vLLM 里测和 fp32 参考的 KL。在独立的确定性引擎里测过（15 位、不截断）：平均 KL 4.1e-4，bf16 推理是 1.5e-3。14 位截断的精度还没单独测。
- **prefill 速度**：只测了 decode。
- **多卡 / 张量并行**、其他模型结构：没测。
- **不切段**会让同时 prefill 很多长 prompt 时的显存压力变大。更好的办法是让 GDN 按绝对位置固定分块，这要改 vLLM 的 GDN kernel。
- 大批量时还剩约 30% 的开销：批 256 时矩阵乘接近算力瓶颈，3 次 INT8 dot 理论上是 bf16 的 1.5 倍，这个 Triton kernel 实测 1.9 倍。

## 文件

```
src/vllm_exact/__init__.py         enable()：设置、绕过 GDN 检查、注册插件、装注意力补丁
src/vllm_exact/plugin.py           quantization="exact_int8"（线性层 + lm_head）
src/vllm_exact/kernels.py          数位化、精确矩阵乘（含 split-K）的 Triton 实现和 torch 参考
src/vllm_exact/paged_attn.py       分页 bf16x2 decode 注意力 + FlashAttentionImpl 补丁
src/vllm_exact/cfg_table_h200_F14.json   H200 调参表
scripts/check_invariance.py        批组成无关 + 测速
scripts/tune_table.py              生成调参表
tests/                             CPU / GPU 单元测试
```
