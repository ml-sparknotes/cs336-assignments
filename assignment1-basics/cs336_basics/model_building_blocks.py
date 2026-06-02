import token
from numpy import dtype
from sympy.polys.polyconfig import query
import torch
import math
from einops import rearrange, einsum, reduce, repeat


def _init_params(shape, normalizer, device, dtype):
    """Initialize a parameter tensor with truncated normal distribution using Glorot-style variance.

    Args:
        shape: Shape of the parameter tensor to create.
        normalizer: Sum of fan-in and fan-out, used to compute variance as 2/normalizer.
        device: Device on which to create the tensor.
        dtype: Data type of the tensor.

    Returns:
        A torch.nn.Parameter initialized with a truncated normal distribution
        (mean=0, std=sqrt(2/normalizer), clipped at ±3*std).
    """
    variance = 2./(normalizer)
    std = math.sqrt(variance)
    holder = torch.zeros(shape, device=device, dtype=dtype)
    return torch.nn.Parameter(torch.nn.init.trunc_normal_(holder, mean=0.0, std=std, a=-3*std, b=3*std))

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self._weights = _init_params((out_features, in_features), in_features + out_features, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self._weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self._embeddings = _init_params((num_embeddings, embedding_dim), embedding_dim, device, dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embeddings[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self._gain = _init_params((d_model,), d_model, device, dtype)
        self._eps = eps
        self._d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sum_sq = einsum(x, x, "... d_model, ... d_model -> ...")
        rms = torch.sqrt(sum_sq / self._d_model + self._eps)
        rms = rearrange(rms, '... -> ... 1')
        result = x * self._gain / rms

        # Return the result in the original dtype
        return result.to(in_dtype)

class PositionWiseFF(torch.nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        self._d_model = d_model
        self._d_ff = 64 * round((8.*d_model/3) / 64) if not d_ff else d_ff
        self._l1 = _init_params((self._d_ff, self._d_model), d_model, device, dtype)
        self._l1_gate = _init_params((self._d_ff, self._d_model), d_model, device, dtype)
        self._l2 = _init_params((self._d_model, self._d_ff), d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_1 = einsum(x, self._l1, "... d_model, d_ff d_model -> ... d_ff")
        gating_input = einsum(x, self._l1_gate, "... d_model, d_ff d_model -> ... d_ff")
        gated_output = gating_input * torch.sigmoid(gating_input) * linear_1
        output = einsum(gated_output, self._l2, "... d_ff, d_model d_ff -> ... d_model")
        return output 

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        seq = torch.arange(max_seq_len)
        dims = torch.arange(0, d_k, 2)
        seq = rearrange(seq, "dims -> dims 1")
        dims = rearrange(dims, "dims -> 1 dims")

        angles = seq/torch.pow(theta, dims / d_k)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        row_1 = torch.stack((cos, -sin), dim=-1)
        row_2 = torch.stack((sin, cos), dim=-1)

        rotation_matrix = torch.stack((row_1, row_2), dim=-2)
        self.register_buffer('rotation_matrix', rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotation_matrix = self.rotation_matrix[token_positions]
        x = rearrange(x, "... seq_len (d two) -> ... seq_len d two", two=2)
        result = einsum(rotation_matrix, x, "... dims row col, ... dims col -> ... dims row")
        result = rearrange(result, "... seq_len d two -> ... seq_len (d two)")
        return result


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1000, apply_rope: bool = True, rope_theta: int = 10000, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_h = d_model // num_heads
        self._q_proj = Linear(d_model, d_model, device, dtype)
        self._k_proj = Linear(d_model, d_model, device, dtype)
        self._v_proj = Linear(d_model, d_model, device, dtype)
        self._output_layer = Linear(d_model, d_model, device, dtype)
        self._apply_rope = apply_rope
        self._rope = RotaryPositionalEmbedding(rope_theta, self._d_h, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q = rearrange(self._q_proj(x), "... seq_len (num_heads dims) -> ... num_heads seq_len dims", num_heads=self._num_heads)
        k = rearrange(self._k_proj(x), "... seq_len (num_heads dims) -> ... num_heads seq_len dims", num_heads=self._num_heads)
        v = rearrange(self._v_proj(x), "... seq_len (num_heads dims) -> ... num_heads seq_len dims", num_heads=self._num_heads)

        if self._apply_rope:
            if token_positions is None:
                token_positions = repeat(torch.arange(x.shape[-2], device=x.device), 'd -> b num_heads d', b=x.shape[0], num_heads=self._num_heads)
            q = self._rope(q, token_positions)
            k = self._rope(k, token_positions)

        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(q, k, v, mask)
        attn_output = rearrange(attn_output, "... num_heads seq_len d_h -> ... seq_len (num_heads d_h)")
        return self._output_layer(attn_output)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta, device=None, dtype=None):
        super().__init__()
        self._mha = CausalMultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, rope_theta=rope_theta, device=device,
        )
        self._rms1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self._rms2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self._ff = PositionWiseFF(
            d_model=d_model, d_ff=d_ff, device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._mha(self._rms1(x))
        return x + self._ff(self._rms2(x))


class TransformerLM(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta, num_layers, vocab_size, device=None, dtype=None):
        super().__init__()
        self._transformer_layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self._embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self._post_norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self._linear = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding(x)
        for layer in self._transformer_layers:
            x = layer(x)
        x = self._post_norm(x)
        x = self._linear(x)
        return x


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_max = torch.max(x, dim=i).values.unsqueeze(dim=i)
    x = x - x_max
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=i).unsqueeze(dim=i)

def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    k, q, v: b, ..., seq, d
    """
    scores = einsum(queries, keys, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    scores = scores / math.sqrt(keys.shape[-1])
    if mask is not None:
        scores = torch.where(mask, scores, -torch.inf)
    probs = softmax(scores, -1)
    values = einsum(probs, values, "... seq_q seq_k, ... seq_k d -> ... seq_q d")
    return values

def model_size_in_bytes(model: torch.nn.Module):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size

def model_size_in_mb(model):
    return model_size_in_bytes(model) / (1024 ** 2)

def cross_entropy_from_logits(inputs, targets):
    per_dim_max = inputs.max(dim=-1).values.unsqueeze(-1)
    normalizer = torch.exp(inputs - per_dim_max).sum(dim=-1)
    selected = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - per_dim_max.squeeze(-1)
    return -(selected - torch.log(normalizer)).mean()
