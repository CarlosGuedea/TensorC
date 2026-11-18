#!/usr/bin/env python3
"""Code-LLM pequeño (~27M parámetros) entrenado con TensorC puro.

GPT de bolsillo especializado en programación: corpus real de Python
(codeparrot-clean, ~200 MB muestreados) y Bash (scripts de repos públicos
de GitHub), tokenizer BPE de nivel byte (~8k tokens), bloques transformer
pre-norm causales, Adam con warmup + decaimiento coseno, checkpoints y
generación con temperatura.

Usa las ops de lenguaje del núcleo: embedding (gather sobre tabla),
cross_entropy_sparse (objetivos enteros, sin one-hot) y Adam.set_lr para
el schedule de learning rate.

Uso:
    python examples/code_llm.py                     # corrida estándar
    python examples/code_llm.py --smoke             # prueba corta (300 pasos)
    python examples/code_llm.py --steps 8000 --batch 16
    python examples/code_llm.py --resume benchmarks/data/code_llm.ckpt
    python examples/code_llm.py --generate "def fibonacci(n):" --gen-len 300
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import tensorc as tc

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "benchmarks", "data")
PY_CORPUS = os.path.join(DATA_DIR, "python.jsonl")
SH_CORPUS = os.path.join(DATA_DIR, "shell.txt")
TOK_JSON = os.path.join(DATA_DIR, "code_tokenizer.json")
TOK_NPY = os.path.join(DATA_DIR, "code_tokens.npy")
CKPT = os.path.join(DATA_DIR, "code_llm.ckpt")
CONFIG_JSON = os.path.join(DATA_DIR, "code_llm_config.json")

SEED = 7


# ------------------------------------------------------------------ datos
def load_documents(max_docs, py_bytes_budget):
    """Devuelve (documentos_python, documentos_shell) muestreados."""
    py_docs = []
    total = 0
    with open(PY_CORPUS, encoding="utf-8") as f:
        for line in f:
            if len(py_docs) >= max_docs or total >= py_bytes_budget:
                break
            try:
                content = json.loads(line).get("content", "")
            except json.JSONDecodeError:
                continue  # la última línea puede venir truncada del stream
            if len(content) >= 100:
                py_docs.append(content)
                total += len(content)
    sh_docs = []
    if os.path.exists(SH_CORPUS):
        with open(SH_CORPUS, encoding="utf-8", errors="replace") as f:
            text = f.read()
        # Trocea la concatenación de scripts en documentos de ~4 KB.
        for start in range(0, len(text), 4000):
            chunk = text[start:start + 4000]
            if len(chunk) >= 200:
                sh_docs.append(chunk)
    return py_docs, sh_docs


def build_tokenizer(docs, vocab_size):
    """Entrena (o carga de caché) un BPE de nivel byte estilo GPT-2."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if os.path.exists(TOK_JSON):
        return Tokenizer.from_file(TOK_JSON)
    print("entrenando tokenizer BPE (vocab %d)..." % vocab_size)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    # Muestra acotada para que el entrenamiento del BPE sea rápido.
    sample = [d for d in docs[:4000]]
    tok.train_from_iterator(sample, trainer)
    tok.save(TOK_JSON)
    return tok


def encode_corpus(py_docs, sh_docs, tok, eos_id):
    """Codifica el corpus a un array uint16 de ids (con caché en disco)."""
    if os.path.exists(TOK_NPY):
        return np.load(TOK_NPY)
    print("codificando corpus (%d docs python, %d docs shell)..."
          % (len(py_docs), len(sh_docs)))
    t0 = time.perf_counter()
    parts = []
    for docs in (py_docs, sh_docs):
        for enc in tok.encode_batch(docs):
            parts.append(np.asarray(enc.ids + [eos_id], dtype=np.uint16))
    tokens = np.concatenate(parts)
    np.save(TOK_NPY, tokens)
    print("  %d tokens en %.1f s" % (len(tokens), time.perf_counter() - t0))
    return tokens


# ------------------------------------------------------------------ modelo
def block_forward(x, p, heads):
    """Bloque pre-norm causal. p = [Wq,Wk,Wv,Wo,W1,b1,W2,b2]."""
    b, s, d = (int(v) for v in x.shape)
    flat = (b * s, d)
    x2 = x.reshape(flat)
    ln1 = x2.layer_norm()
    q = ln1.matmul(p[0]).reshape((b, s, d))
    k = ln1.matmul(p[1]).reshape((b, s, d))
    v = ln1.matmul(p[2]).reshape((b, s, d))
    attn = q.multihead_attention(k, v, heads, 1)
    proj = attn.reshape(flat).matmul(p[3])
    h1 = x2.add(proj)
    ln2 = h1.layer_norm()
    ff1 = ln2.matmul(p[4]).add(p[5]).relu()
    ff3 = ff1.matmul(p[6]).add(p[7])
    return h1.add(ff3).reshape((b, s, d))


class CodeGPT:
    def __init__(self, vocab, dim, ff, heads, blocks, seq):
        rng = np.random.default_rng(SEED)
        self.vocab, self.dim, self.seq, self.heads = vocab, dim, seq, heads
        self.embed = tc.Tensor(
            (rng.standard_normal((vocab, dim)) * 0.02).astype(np.float32),
            requires_grad=True)
        self.blocks = [tc.TransformerBlock(dim, ff, heads, seed=SEED + 1 + i)
                       for i in range(blocks)]
        self.w_out = tc.Tensor(
            (rng.standard_normal((dim, vocab)) * 0.02).astype(np.float32),
            requires_grad=True)
        self.b_out = tc.Tensor(np.zeros((1, vocab), dtype=np.float32),
                               requires_grad=True)
        self.params = ([self.embed] +
                       [p for blk in self.blocks for p in blk.params] +
                       [self.w_out, self.b_out])
        self.pos = tc.Tensor(np.asarray(
            tc.sinusoidal_positions(seq, dim).data, dtype=np.float32)
            .reshape(seq, dim))

    def forward(self, ids):
        """ids tc.Tensor [B, S] (floats enteros) -> probs [B*S, vocab]."""
        b, s = (int(v) for v in ids.shape)
        h = self.embed.embedding(ids).add(self.pos)
        for blk in self.blocks:
            h = block_forward(h, blk.params, self.heads)
        h2 = h.reshape((b * s, self.dim)).layer_norm()
        return h2.matmul(self.w_out).add(self.b_out).softmax()


def count_params(model):
    return sum(int(np.prod(p.shape)) for p in model.params)


# --------------------------------------------------------------- LR cosine
def make_lr_schedule(peak, floor, warmup, total):
    def lr_at(step):
        if step < warmup:
            return peak * (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * t))
    return lr_at


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Code-LLM pequeño con TensorC")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--py-budget", type=int, default=80000000,
                    help="tope de bytes de texto Python a usar")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--generate", type=str, default=None,
                    help="solo generar a partir de este prompt")
    ap.add_argument("--gen-len", type=int, default=250)
    ap.add_argument("--smoke", action="store_true",
                    help="corrida corta de humo: 300 pasos, batch 8")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.sample_every, args.save_every = 300, 100, 300

    print("backend CUDA: %s" % ("activo" if tc.cuda_available()
                                  else "NO DISPONIBLE (CPU)"))
    py_docs, sh_docs = load_documents(args.max_docs, args.py_budget)
    print("corpus: %d docs python (%.1f MB), %d docs shell (%.1f MB)"
          % (len(py_docs), sum(len(d) for d in py_docs) / 1e6,
             len(sh_docs), sum(len(d) for d in sh_docs) / 1e6))

    tok = build_tokenizer(py_docs + sh_docs, args.vocab)
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id("<|endoftext|>")
    tokens = encode_corpus(py_docs, sh_docs, tok, eos_id)
    print("tokens totales: %d | vocab: %d" % (len(tokens), vocab))

    model = CodeGPT(vocab, args.dim, args.ff, args.heads, args.layers,
                    args.seq)
    n_params = count_params(model)
    print("modelo: dim %d, %d bloques, %d cabezas, ff %d, seq %d -> "
          "%.2fM parámetros" % (args.dim, args.layers, args.heads, args.ff,
                                args.seq, n_params / 1e6))

    if args.resume and os.path.exists(args.resume):
        saved = tc.load(args.resume, requires_grad=True)
        if isinstance(saved, tc.Tensor):
            saved = [saved]
        assert len(saved) == len(model.params), \
            "checkpoint con %d tensores; el modelo tiene %d" \
            % (len(saved), len(model.params))
        for p, s in zip(model.params, saved):
            assert list(p.shape) == list(s.shape), \
                "forma distinta: %s vs %s" % (p.shape, s.shape)
            p.copy_(s)
        print("reanudado desde %s" % args.resume)

    opt = tc.Adam(model.params, args.lr)

    def generate(prompt, length, temperature=0.8, greedy=False):
        ids = tok.encode(prompt).ids[-args.seq:]
        rng = np.random.default_rng(SEED + 1)
        for _ in range(length):
            ctx = ([eos_id] * (args.seq - len(ids)) + ids)[-args.seq:]
            xb = np.asarray([ctx], dtype=np.float32)
            probs = model.forward(tc.Tensor(xb))
            last = probs.slice_axis(0, args.seq - 1, 1)
            p = np.asarray(last.data, dtype=np.float64).reshape(-1)
            if greedy:
                nxt = int(p.argmax())
            else:
                p = np.log(p + 1e-12) / temperature
                p = np.exp(p - p.max())
                nxt = int(rng.choice(vocab, p=p / p.sum()))
            ids.append(nxt)
            if nxt == eos_id:
                break
        return tok.decode(ids)

    if args.generate is not None:
        print(generate(args.generate, args.gen_len))
        return

    # ---------------------------------------------------- entrenamiento
    n = len(tokens) - args.seq - 1
    rng = np.random.default_rng(SEED)
    lr_at = make_lr_schedule(args.lr, args.lr / 10.0, args.warmup,
                             args.steps)
    print("entrenando %d pasos (batch %d, seq %d, %.1fK tokens/paso, "
          "lr pico %.1e con warmup %d)..."
          % (args.steps, args.batch, args.seq,
             args.batch * args.seq / 1000.0, args.lr, args.warmup))
    t0 = time.perf_counter()
    losses = []
    for step in range(args.steps):
        starts = rng.integers(0, n, size=args.batch)
        xb = np.stack([tokens[s:s + args.seq] for s in starts]
                      ).astype(np.float32)
        yb = np.stack([tokens[s + 1:s + args.seq + 1] for s in starts]
                      ).astype(np.float32).reshape(-1)
        probs = model.forward(tc.Tensor(xb))
        loss = probs.cross_entropy_sparse(tc.Tensor(yb))
        loss.backward()
        opt.set_lr(lr_at(step))
        opt.step()
        losses.append(float(loss.data[0]))
        if (step + 1) % 100 == 0 or step == 0:
            recent = float(np.mean(losses[-100:]))
            print("  paso %5d/%d  loss %.4f  lr %.2e  (%.0f ms/paso)"
                  % (step + 1, args.steps, recent, lr_at(step),
                     (time.perf_counter() - t0) / (step + 1) * 1000))
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            tc.save(model.params, CKPT)
            with open(CONFIG_JSON, "w", encoding="utf-8") as f:
                json.dump({"vocab": vocab, "dim": args.dim, "ff": args.ff,
                           "heads": args.heads, "layers": args.layers,
                           "seq": args.seq, "step": step + 1}, f)
        if (step + 1) % args.sample_every == 0:
            print("  --- muestra en el paso %d ---" % (step + 1))
            print(generate("def ", 120, temperature=0.8))
            print("  " + "-" * 50)

    total = (time.perf_counter() - t0) * 1000.0
    print("entrenamiento total: %.0f ms (%.0f ms/paso) | loss media "
          "inicial %.3f -> final %.3f"
          % (total, total / args.steps,
             np.mean(losses[:50]), np.mean(losses[-50:])))
    print("checkpoint: %s" % CKPT)

    print("\n--- muestras finales (temperature 0.8) ---")
    for prompt in ("def fibonacci(n):", "#!/bin/bash\n", "import numpy"):
        print(generate(prompt, 200, temperature=0.8))
        print("-" * 60)


if __name__ == "__main__":
    main()
