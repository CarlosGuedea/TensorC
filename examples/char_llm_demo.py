#!/usr/bin/env python3
"""LLM miniature a nivel carácter con TensorC puro (GPT de bolsillo).

Demuestra que las piezas de TensorC componen un modelo tipo GPT:
embeddings vía one-hot x matriz, bloques transformer pre-norm con máscara
causal, posiciones sinusoidales, cabeza de logits con softmax +
cross-entropy, Adam y muestreo con temperatura.

Entrena con el Quijote (Project Gutenberg, dominio público) descargado
automáticamente, e imprime la curva de pérdida y texto generado.

Uso:
    python examples/char_llm_demo.py [pasos]
"""

import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import tensorc as tc

DATA = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data",
                    "quijote.txt")
TEXT_URL = ("https://github.com/juanpampliega/datasets/raw/master/"
            "don-quijote.txt.gz")

DIM = 64          # dimensión del modelo
FF = 128          # dimensión de la FFN
HEADS = 2         # cabezas de atención
BLOCKS = 2        # bloques transformer
SEQ = 64          # longitud de contexto
BATCH = 48        # ventanas por paso
LR = 1e-3
SEED = 7


def load_text():
    if not os.path.exists(DATA):
        print("descargando El Quijote (Project Gutenberg)...")
        import gzip
        import urllib.request
        with urllib.request.urlopen(TEXT_URL) as response:
            with gzip.GzipFile(fileobj=response) as archive:
                with open(DATA, "wb") as out:
                    out.write(archive.read())
    with open(DATA, encoding="utf-8-sig") as f:
        text = f.read()
    start = text.find("En un lugar de la Mancha")
    return text[start:start + 1200000]  # ~1.2 M de caracteres


def build_vocab(text):
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    chars = sorted(c for c, n in counts.items() if n >= 5)
    if "\n" not in chars:
        chars.append("\n")
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = <unk>
    itos = {i + 1: c for i, c in enumerate(chars)}
    itos[0] = "?"
    return stoi, itos, len(itos)


def block_forward(x, p, heads):
    """Bloque pre-norm causal (igual que tc_transformer_block pero con
    máscara causal en la atención). p = [Wq,Wk,Wv,Wo,W1,b1,W2,b2]."""
    b, s, d = x.shape
    flat = (b * s, d)
    x2 = x.reshape(flat)
    ln1 = x2.layer_norm()
    q = (ln1.matmul(p[0])).reshape((b, s, d))
    k = (ln1.matmul(p[1])).reshape((b, s, d))
    v = (ln1.matmul(p[2])).reshape((b, s, d))
    attn = q.multihead_attention(k, v, heads, 1)
    proj = attn.reshape(flat).matmul(p[3])
    h1 = x2.add(proj)
    ln2 = h1.layer_norm()
    ff1 = ln2.matmul(p[4]).add(p[5]).relu()
    ff3 = ff1.matmul(p[6]).add(p[7])
    return h1.add(ff3).reshape((b, s, d))


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rng = np.random.default_rng(SEED)
    text = load_text()
    stoi, itos, vocab = build_vocab(text)
    tokens = np.array([stoi.get(c, 0) for c in text], dtype=np.int64)
    print("caracteres: %d | vocabulario: %d | modelo: %d bloques, dim %d, "
          "%d cabezas, seq %d, batch %d"
          % (len(tokens), vocab, BLOCKS, DIM, HEADS, SEQ, BATCH))

    # Parámetros: embedding, bloques, cabeza de salida.
    scale = lambda fan_in, fan_out: 0.02
    E = tc.Tensor((rng.standard_normal((vocab, DIM)) * 0.1).astype(np.float32),
                  requires_grad=True)
    blocks = [tc.TransformerBlock(DIM, FF, HEADS, seed=SEED + i)
              for i in range(BLOCKS)]
    W_out = tc.Tensor((rng.standard_normal((DIM, vocab)) * 0.02).astype(np.float32),
                      requires_grad=True)
    b_out = tc.Tensor(np.zeros((1, vocab), dtype=np.float32), requires_grad=True)
    params = [E] + [p for blk in blocks for p in blk.params] + [W_out, b_out]
    opt = tc.Adam(params, LR)

    eye = np.eye(vocab, dtype=np.float32)
    pos = tc.Tensor(np.asarray(
        tc.sinusoidal_positions(SEQ, DIM).data, dtype=np.float32)
        .reshape(SEQ, DIM))

    def forward(tok_ids):
        """tok_ids [B, S] -> logits softmax [B*S, vocab]."""
        b = len(tok_ids)
        one_hot = eye[tok_ids].reshape(b * SEQ, vocab)
        h = tc.Tensor(one_hot).matmul(E).reshape((b, SEQ, DIM)).add(pos)
        for blk, p in zip(blocks, [blk.params for blk in blocks]):
            h = block_forward(h, p, HEADS)
        h2 = h.reshape((b * SEQ, DIM)).layer_norm()
        return h2.matmul(W_out).add(b_out).softmax()

    # ------------------------------------------------------ entrenamiento
    n = len(tokens) - SEQ - 1
    starts_all = rng.integers(0, n, size=steps * BATCH)
    print("entrenando %d pasos..." % steps)
    t0 = time.perf_counter()
    losses = []
    for step in range(steps):
        starts = starts_all[step * BATCH:(step + 1) * BATCH]
        xb = np.stack([tokens[s:s + SEQ] for s in starts])
        yb = np.stack([tokens[s + 1:s + SEQ + 1] for s in starts])
        probs = forward(xb)
        loss = probs.cross_entropy(tc.Tensor(eye[yb.reshape(-1)]))
        loss.backward()
        opt.step()
        losses.append(float(loss.data[0]))
        if (step + 1) % 100 == 0:
            recent = np.mean(losses[-100:])
            print("  paso %4d/%d  loss %.4f  (%.0f ms/paso)"
                  % (step + 1, steps, recent,
                     (time.perf_counter() - t0) / (step + 1) * 1000))
    total = (time.perf_counter() - t0) * 1000.0
    print("entrenamiento total: %.0f ms | loss inicial %.3f -> final %.3f"
          % (total, np.mean(losses[:50]), np.mean(losses[-50:])))

    # ---------------------------------------------------------- muestreo
    def generate(prompt, length=220, temperature=0.8):
        ids = [stoi.get(c, 0) for c in prompt][-SEQ:]
        out = prompt
        for _ in range(length):
            ctx = np.array([([0] * (SEQ - len(ids)) + ids)[-SEQ:]],
                           dtype=np.int64)
            probs = forward(ctx)
            last = probs.slice_axis(0, SEQ - 1, 1)
            p = np.asarray(last.data, dtype=np.float64).reshape(-1)
            p = np.log(p + 1e-12) / temperature
            p = np.exp(p - p.max())
            nxt = rng.choice(vocab, p=p / p.sum())
            ids.append(nxt)
            out += itos[nxt]
        return out

    print("\n--- muestras (temperature 0.8) ---")
    for prompt in ("En un lugar de la Mancha, ",
                   "Don Quijote ",
                   "Sancho Panza "):
        print(generate(prompt))
        print("-" * 60)


if __name__ == "__main__":
    main()
