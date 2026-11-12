"""vLLM 量化插件 quantization="exact_int8"：所有线性层（可选含 lm_head）换成 INT8 两数位精确累加。

不改 vLLM 源码：register_quantization_config + torch custom op。由 vllm_exact.enable() 导入并配置。
GDN 的卷积、embedding 查表不走线性层的 apply，保持原样。
"""
import json
from pathlib import Path

import torch

from . import kernels as K
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

# 由 enable() 设置
F = 14
TRUNC = True
LMHEAD = True
_TABLE = None
RB = 16384  # 按行分块：每行独立，结果不变；临时显存有上界


def load_table(path):
    """调参表：{"N,K": {"M": [BM, BN, BK, warps, stages, GM, SK]}}。换配置不改变结果，只影响速度。"""
    global _TABLE
    if path is None:
        _TABLE = None
        return
    t = json.loads(Path(path).read_text())
    _TABLE = {tuple(map(int, k.split(","))): {int(m): tuple(c) for m, c in v.items()} for k, v in t.items()}


def _cfg(M, N, Kd):
    if _TABLE is not None and (N, Kd) in _TABLE:
        t = _TABLE[(N, Kd)]
        ms = sorted(t)
        if M > ms[-1]:  # 超出调参范围（prefill）：不用 split-K，免得部分和的临时显存太大
            return tuple(t[ms[-1]][:6]) + (1,)
        return t[next(x for x in ms if x >= M)]  # 不小于 M 的最近档位
    if M <= 16:
        return (16, 128, 128, 4, 4, 8, 1)
    if M <= 64:
        return (64, 128, 128, 4, 4, 8, 1)
    return (128, 128, 128, 8, 3, 8, 1)


@torch.library.custom_op("exactmm::linear", mutates_args=())
def exact_linear_op(x: torch.Tensor, wh: torch.Tensor, wl: torch.Tensor, cw: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    shp = x.shape
    N = wh.shape[0]
    x2 = x.reshape(-1, shp[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, Kd = x2.shape
    out = x.new_empty(M, N)
    for i in range(0, M, RB):
        xb = x2[i:i + RB]
        xd = K.digitize(xb, F)  # 直接读 bf16（bf16 → fp32 精确，数位不变）
        K.exact_linear(xd, (wh, wl, cw, sw), F, TRUNC, out=out[i:i + RB], cfg=_cfg(xb.shape[0], N, Kd))  # 直接写 bf16
    return out.reshape(*shp[:-1], N)


@exact_linear_op.register_fake
def _(x, wh, wl, cw, sw):
    return x.new_empty(*x.shape[:-1], wh.shape[0])


def _digitize_layer(layer):
    w = layer.weight.data
    parts = [K.digitize(w[i:i + 32768].float().contiguous(), F) for i in range(0, w.shape[0], 32768)]  # 按行分块，避免一次转 fp32
    h, l, cs, sh = (torch.cat([p[j] for p in parts]) for j in range(4))
    layer.register_buffer("ex_wh", h.contiguous(), persistent=False)
    layer.register_buffer("ex_wl", l.contiguous(), persistent=False)
    layer.register_buffer("ex_cw", cs.contiguous(), persistent=False)
    layer.register_buffer("ex_sw", sh.contiguous(), persistent=False)


def _apply(layer, x, bias):
    y = torch.ops.exactmm.linear(x, layer.ex_wh, layer.ex_wl, layer.ex_cw, layer.ex_sw)
    return y if bias is None else y + bias


class ExactLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _digitize_layer(layer)

    def apply(self, layer, x, bias=None):
        return _apply(layer, x, bias)


class ExactLMHeadMethod(UnquantizedEmbeddingMethod):
    """lm_head 走精确线性层。lm_head 不做 embedding 查表，父类的 embedding() 原样保留。"""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _digitize_layer(layer)

    def apply(self, layer, x, bias=None):
        return _apply(layer, x, bias)


@register_quantization_config("exact_int8")
class ExactInt8Config(QuantizationConfig):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "exact_int8"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 80

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return ExactLinearMethod()
        if LMHEAD and isinstance(layer, ParallelLMHead):
            return ExactLMHeadMethod()
        return None
