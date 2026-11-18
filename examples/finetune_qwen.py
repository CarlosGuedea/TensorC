#!/usr/bin/env python3
"""Fine-tuning PEFT de Qwen2.5-Coder-0.5B con TensorC puro (sin PyTorch).

Congela la base y entrena solo las últimas N capas (attn + mlp + norms) más
la norma final (~59M parámetros con N=4, de 494M totales). Corpus: el mismo
de code_llm (python.jsonl + shell.txt) re-codificado con el tokenizer de
Qwen (vocab 151936, uint32).

El forward/backward corre entero en GPU (rms_norm, matmul cuBLAS, RoPE y
GQA vía matmuls constantes, multihead_attention causal, softmax, sparse CE).

Uso:
    python examples/finetune_qwen.py --smoke            # 30 pasos de prueba
    python examples/finetune_qwen.py --steps 2000 --layers 4
    python examples/finetune_qwen.py --chat             # chatear con el modelo
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from qwen_tensorc import DATA_DIR, load_model

import tensorc as tc
from tokenizers import Tokenizer

TOKENS_NPY = os.path.join(DATA_DIR, "ft_tokens.npy")
ADAPTER = os.path.join(DATA_DIR, "qwen_ft_adapter.ckpt")
ADAPTER_JSON = os.path.join(DATA_DIR, "qwen_ft_adapter.json")
CORPUS_CAP_CHARS = 30_000_000   # ~30 MB de texto bastan para el PEFT
SEED = 7


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def build_tokens(tok):
    """Codifica el corpus con el tokenizer de Qwen y lo cachea en uint32."""
    if os.path.exists(TOKENS_NPY):
        return np.load(TOKENS_NPY)
    texts = []
    pj = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data",
                      "python.jsonl")
    sh = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "data",
                      "shell.txt")
    total = 0
    with open(pj, encoding="utf-8") as f:
        for line in f:
            try:
                content = json.loads(line)["content"]
            except (ValueError, KeyError):
                continue
            texts.append(content)
            total += len(content)
            if total >= CORPUS_CAP_CHARS * 3 // 4:
                break
    with open(sh, encoding="utf-8", errors="ignore") as f:
        texts.append(f.read(CORPUS_CAP_CHARS // 4))
    print(f"corpus: {total / 1e6:.0f} MB de código; codificando con Qwen "
          "BPE...")
    t0 = time.perf_counter()
    chunks = []
    for i in range(0, len(texts), 256):
        for enc in tok.encode_batch(texts[i:i + 256]):
            chunks.append(np.asarray(enc.ids, dtype=np.uint32))
            chunks.append(np.asarray([151643], dtype=np.uint32))  # <|endoftext|>
    tokens = np.concatenate(chunks)
    np.save(TOKENS_NPY, tokens)
    print(f"tokens: {len(tokens) / 1e6:.1f}M "
          f"({time.perf_counter() - t0:.0f}s) -> {TOKENS_NPY}")
    return tokens


# --------------------------------------------------------------------------
# Generación
# --------------------------------------------------------------------------

def generate(model, tok, prompt, length=60, temperature=0.2, greedy=False):
    ids = tok.encode(prompt).ids
    rng = np.random.default_rng(SEED + 1)
    for _ in range(length):
        ctx = ids[-model.max_seq:]
        xb = np.asarray([ctx], dtype=np.float32)
        logits = model.forward(tc.Tensor(xb))
        last = logits.slice_axis(0, len(ctx) - 1, 1)
        row = np.asarray(last.data, dtype=np.float32)
        if greedy:
            nxt = int(row.argmax())
        else:
            p = np.exp((row - row.max()) / max(temperature, 1e-5))
            nxt = int(rng.choice(len(p), p=p / p.sum()))
        ids.append(nxt)
        if nxt == 151643:
            break
    return tok.decode(ids)


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (
        1.0 + np.cos(np.pi * t))


def train(args):
    tokens = build_tokens(Tokenizer.from_file(
        os.path.join(DATA_DIR, "tokenizer.json")))
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))

    trainable = {"ln_f"}
    for i in range(args.layers_start, 24):
        trainable |= {f"layers.{i}.attn", f"layers.{i}.mlp",
                      f"layers.{i}.ln1", f"layers.{i}.ln2"}
    print(f"cargando modelo (entrenables: capas {args.layers_start}-23 + "
          "ln_f)...")
    model = load_model(max_seq=args.seq, trainable=trainable)
    n_train = sum(int(np.prod(p.shape)) for p in model.params)
    print(f"parámetros entrenables: {n_train / 1e6:.1f}M de ~494M "
          f"({100.0 * n_train / 494e6:.1f}%)")

    if args.resume and os.path.exists(ADAPTER):
        saved = tc.load(ADAPTER, requires_grad=True)
        for p, s in zip(model.params, saved):
            p.copy_(s)
        print(f"adaptador reanudado desde {ADAPTER}")

    opt = tc.Adam(model.params, args.lr)
    n = len(tokens) - args.seq - 1
    rng = np.random.default_rng(SEED)
    print(f"fine-tune: {args.steps} pasos (batch {args.batch}, seq "
          f"{args.seq}, lr pico {args.lr:.1e})")
    t0 = time.perf_counter()
    losses = []
    for step in range(args.steps):
        starts = rng.integers(0, n, size=args.batch)
        xb = np.stack([tokens[s:s + args.seq] for s in starts]
                      ).astype(np.float32)
        yb = np.stack([tokens[s + 1:s + args.seq + 1] for s in starts]
                      ).astype(np.float32).reshape(-1)
        ids_t = tc.Tensor(xb)
        tgt_t = tc.Tensor(yb)
        loss = model.loss(ids_t, tgt_t)
        loss.backward()
        opt.set_lr(float(lr_at(step, args)))
        opt.step()
        for p in model.params:
            p.zero_grad()
        losses.append(float(loss.data[0]))
        # Suelta el grafo ANTES del siguiente forward: si el viejo sigue
        # vivo durante el nuevo, el pico de VRAM se duplica y las ops
        # caen a CPU en silencio (pasos 4x más lentos).
        del loss, ids_t, tgt_t
        if step == 0 or (step + 1) % 10 == 0:
            recent = float(np.mean(losses[-10:]))
            print(f"  paso {step + 1:5d}/{args.steps}  loss {recent:.4f}  "
                  f"lr {lr_at(step, args):.2e}  "
                  f"({(time.perf_counter() - t0) / (step + 1):.1f} s/paso)",
                  flush=True)
        if (step + 1) % args.sample_every == 0:
            print(f"  --- muestra paso {step + 1} ---")
            print(generate(model, tok, "def fibonacci(n):", 50,
                           greedy=True))
            print("  " + "-" * 50)

    tc.save(model.params, ADAPTER)
    with open(ADAPTER_JSON, "w", encoding="utf-8") as f:
        json.dump({"layers_start": args.layers_start,
                   "param_count": n_train, "steps": args.steps}, f)
    total = time.perf_counter() - t0
    print(f"listo: {total:.0f}s total ({total / args.steps:.1f} s/paso) | "
          f"loss {np.mean(losses[:10]):.3f} -> {np.mean(losses[-10:]):.3f}")
    print(f"adaptador guardado en {ADAPTER}")


def chat(args):
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    layers_start = 20
    if os.path.exists(ADAPTER_JSON):
        with open(ADAPTER_JSON, encoding="utf-8") as f:
            layers_start = json.load(f)["layers_start"]
    trainable = {"ln_f"}
    for i in range(layers_start, 24):
        trainable |= {f"layers.{i}.attn", f"layers.{i}.mlp",
                      f"layers.{i}.ln1", f"layers.{i}.ln2"}
    model = load_model(max_seq=args.seq, trainable=trainable)
    if os.path.exists(ADAPTER):
        saved = tc.load(ADAPTER, requires_grad=True)
        for p, s in zip(model.params, saved):
            p.copy_(s)
        print(f"(adaptador cargado: {ADAPTER})")
    print("chat con Qwen2.5-Coder-0.5B (TensorC). Escribe 'salir' para "
          "terminar.")
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("salir", "exit", "quit"):
            break
        if not prompt.strip():
            continue
        print(generate(model, tok, prompt, length=args.gen_len,
                       temperature=0.2))
    print("adiós")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4,
                    help="número de capas finales entrenables")
    ap.add_argument("--sample-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--gen-len", type=int, default=80)
    args = ap.parse_args()
    args.layers_start = 24 - args.layers
    if args.smoke:
        args.steps = 30
        args.sample_every = 15
    if args.chat:
        chat(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
