"""Qwen2.5-Coder-0.5B ensamblado íntegramente con TensorC (sin PyTorch).

Carga los pesos bf16 de model.safetensors (convertidos a fp32), construye el
grafo con ops de TensorC (rms_norm, matmul, multihead_attention causal,
broadcast mul/add) y expone forward() para validación y fine-tuning.

Piezas de arquitectura:
  - RMSNorm (eps 1e-6) con gamma como mul broadcast.
  - RoPE con convención rotate_half de Qwen: out = x*cos + (x @ M)*sin, donde
    M [64,64] es la permutación constante que produce [-x2, x1]; cos/sin se
    duplican a ancho 64 para evitar slice/concat (que no tienen ruta CUDA).
  - GQA (14 Q heads, 2 KV heads): expansión de KV con matriz constante
    E [128, 896] (un 1 por columna), de modo que k_exp = k @ E es un matmul
    más (todo en GPU, con gradiente hacia k).
  - SwiGLU: silu(x @ Wg) * (x @ Wu) @ Wd, con silu(g) = g * sigmoid(g).
  - lm_head atado al embedding: logits = h @ E^T.

Las proyecciones se guardan ya transpuestas [in, out] para usar matmul
directo (x @ W) sin transponer en cada forward.
"""

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import tensorc as tc

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                         "benchmarks", "data", "qwen"))


# --------------------------------------------------------------------------
# Loader safetensors (bf16 -> fp32)
# --------------------------------------------------------------------------

def load_safetensors(path):
    """Devuelve dict nombre -> np.ndarray float32 con los tensores del archivo."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
        raw = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        buf = raw[start:end]
        if dtype == "BF16":
            u16 = np.frombuffer(buf, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(buf, dtype=np.float32).copy()
        elif dtype == "F16":
            arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError(f"dtype no soportado: {dtype} ({name})")
        tensors[name] = arr.reshape(info["shape"])
    return tensors


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

def _leaf(arr, key, trainable, params, reshape=None):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if reshape is not None:
        a = a.reshape(reshape)
    want = key in trainable
    t = tc.Tensor(a, requires_grad=want)
    if want:
        params.append(t)
    return t


class QwenTensorC:
    def __init__(self, weights, config, max_seq=512, trainable=None):
        """weights: dict de safetensors; config: dict de config.json.

        trainable: conjunto de claves con requires_grad=True; el resto queda
        congelado. Claves: "embed", "ln_f", y por capa "layers.N.attn",
        "layers.N.mlp", "layers.N.ln1", "layers.N.ln2".
        """
        self.cfg = config
        self.hidden = config["hidden_size"]
        self.layers_n = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = self.hidden // self.heads
        self.inter = config["intermediate_size"]
        self.vocab = config["vocab_size"]
        self.max_seq = max_seq
        trainable = trainable or set()
        self.params = []
        self._rope_cache = {}

        # Embedding (también sirve de lm_head: logits = h @ E^T).
        self.embed = _leaf(weights["model.embed_tokens.weight"], "embed",
                           trainable, self.params)
        self.ln_f = _leaf(weights["model.norm.weight"], "ln_f", trainable,
                          self.params, reshape=(1, 1, -1))

        self.layers = []
        for i in range(self.layers_n):
            p = f"model.layers.{i}."
            attn_key = f"layers.{i}.attn"
            mlp_key = f"layers.{i}.mlp"
            self.layers.append({
                "ln1": _leaf(weights[p + "input_layernorm.weight"],
                             f"layers.{i}.ln1", trainable, self.params,
                             reshape=(1, 1, -1)),
                "qw": _leaf(weights[p + "self_attn.q_proj.weight"].T,
                            attn_key, trainable, self.params),
                "qb": _leaf(weights[p + "self_attn.q_proj.bias"],
                            attn_key, trainable, self.params,
                            reshape=(1, -1)),
                "kw": _leaf(weights[p + "self_attn.k_proj.weight"].T,
                            attn_key, trainable, self.params),
                "kb": _leaf(weights[p + "self_attn.k_proj.bias"],
                            attn_key, trainable, self.params,
                            reshape=(1, -1)),
                "vw": _leaf(weights[p + "self_attn.v_proj.weight"].T,
                            attn_key, trainable, self.params),
                "vb": _leaf(weights[p + "self_attn.v_proj.bias"],
                            attn_key, trainable, self.params,
                            reshape=(1, -1)),
                "ow": _leaf(weights[p + "self_attn.o_proj.weight"].T,
                            attn_key, trainable, self.params),
                "ln2": _leaf(weights[p + "post_attention_layernorm.weight"],
                             f"layers.{i}.ln2", trainable, self.params,
                             reshape=(1, 1, -1)),
                "gate": _leaf(weights[p + "mlp.gate_proj.weight"].T,
                              mlp_key, trainable, self.params),
                "up": _leaf(weights[p + "mlp.up_proj.weight"].T,
                            mlp_key, trainable, self.params),
                "down": _leaf(weights[p + "mlp.down_proj.weight"].T,
                              mlp_key, trainable, self.params),
            })

        # Matriz constante de rotate_half: rot[:d2] = -x[d2:], rot[d2:] = x[:d2]
        # (rot = x @ M con M[k, j]: M[d2+j, j] = -1, M[j, d2+j] = 1).
        d = self.head_dim
        d2 = d // 2
        M = np.zeros((d, d), dtype=np.float32)
        for j in range(d2):
            M[d2 + j, j] = -1.0
            M[j, d2 + j] = 1.0
        self.rot_m = tc.Tensor(M)

        # Expansión GQA: E [kv_heads*dim, hidden] con E[(h//g)*dim+j, h*dim+j] = 1.
        kv_width = self.kv_heads * d
        group = self.heads // self.kv_heads
        E = np.zeros((kv_width, self.hidden), dtype=np.float32)
        for h in range(self.heads):
            src = (h // group) * d
            for j in range(d):
                E[src + j, h * d + j] = 1.0
        self.expand_kv = tc.Tensor(E)  # k_exp = k @ E

    # -- RoPE ---------------------------------------------------------------

    def _rope_tables(self, seq):
        """cos/sin [1, T, 1, head_dim] duplicados a ancho completo (cached)."""
        cached = self._rope_cache.get(seq)
        if cached:
            return cached
        d = self.head_dim
        d2 = d // 2
        theta = float(self.cfg.get("rope_theta", 1e6))
        inv = 1.0 / (theta ** (np.arange(0, d2, dtype=np.float32) * 2.0 / d))
        t = np.arange(seq, dtype=np.float32)
        freqs = np.outer(t, inv)                    # [T, d2]
        cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=1)
        sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=1)
        cos4 = tc.Tensor(np.ascontiguousarray(cos.reshape(1, seq, 1, d)))
        sin4 = tc.Tensor(np.ascontiguousarray(sin.reshape(1, seq, 1, d)))
        self._rope_cache[seq] = (cos4, sin4)
        return cos4, sin4

    def _rope(self, x4, cos4, sin4, batch, seq, heads):
        """x4: [B, T, H, 64] -> RoPE. rot = (flat @ M) reshaped a x4."""
        flat = x4.reshape((batch * seq * heads, self.head_dim))
        rot = flat.matmul(self.rot_m).reshape((batch, seq, heads,
                                               self.head_dim))
        return x4.mul(cos4).add(rot.mul(sin4))

    # -- Forward --------------------------------------------------------------

    def forward(self, ids):
        """ids: tc.Tensor [B, T] con índices enteros (float). Devuelve los
        logits [B*T, vocab]."""
        B, T = ids.shape
        H, d = self.hidden, self.head_dim
        cos4, sin4 = self._rope_tables(T)
        h3 = self.embed.embedding(ids)              # [B, T, H]
        for lay in self.layers:
            n2 = h3.rms_norm().mul(lay["ln1"]).reshape((B * T, H))
            q = n2.matmul(lay["qw"]).add(lay["qb"])
            k = n2.matmul(lay["kw"]).add(lay["kb"])
            v = n2.matmul(lay["vw"]).add(lay["vb"])
            q4 = self._rope(q.reshape((B, T, self.heads, d)), cos4, sin4,
                            B, T, self.heads)
            k4 = self._rope(k.reshape((B, T, self.kv_heads, d)), cos4, sin4,
                            B, T, self.kv_heads)
            q3 = q4.reshape((B, T, H))
            k3 = k4.reshape((B * T, self.kv_heads * d)) \
                   .matmul(self.expand_kv).reshape((B, T, H))
            v3 = v.reshape((B * T, self.kv_heads * d)) \
                 .matmul(self.expand_kv).reshape((B, T, H))
            att = q3.multihead_attention(k3, v3, self.heads, causal=True)
            a3 = att.reshape((B * T, H)).matmul(lay["ow"]).reshape((B, T, H))
            h3 = h3.add(a3)
            n2 = h3.rms_norm().mul(lay["ln2"]).reshape((B * T, H))
            g = n2.matmul(lay["gate"])
            u = n2.matmul(lay["up"])
            ffn = g.mul(g.sigmoid()).mul(u).matmul(lay["down"]) \
                  .reshape((B, T, H))
            h3 = h3.add(ffn)
        nf = h3.rms_norm().mul(self.ln_f).reshape((B * T, H))
        return nf.matmul(self.embed.transpose())

    def loss(self, ids, targets):
        """ids [B, T], targets [B*T] floats enteros. Cross-entropy media."""
        logits = self.forward(ids)
        return logits.softmax().cross_entropy_sparse(targets)


def load_model(data_dir=DATA_DIR, max_seq=512, trainable=None):
    with open(os.path.join(data_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    weights = load_safetensors(os.path.join(data_dir, "model.safetensors"))
    return QwenTensorC(weights, config, max_seq=max_seq, trainable=trainable)


if __name__ == "__main__":
    from tokenizers import Tokenizer

    model = load_model(max_seq=64)
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    ids = tok.encode("def quicksort(arr):").ids
    x = tc.Tensor(np.array([ids], dtype=np.float32))
    logits = model.forward(x)
    print("logits:", logits.shape)
    row = np.array(logits.data[-logits.shape[1]:])
    top = int(np.argmax(row))
    print("next token:", top, repr(tok.decode([top])))
