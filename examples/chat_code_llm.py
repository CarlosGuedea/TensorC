#!/usr/bin/env python3
"""Chat de terminal con el code-LLM entrenado (ver examples/code_llm.py).

Carga el checkpoint y el tokenizer BPE y abre un REPL: escribes un prompt
(código o comentario) y el modelo lo continúa. Es un modelo base de 27M
entrenado desde cero — completa código con sintaxis plausible de
Python/Bash; no es un asistente instruido.

Comandos dentro del chat:
    /salir            termina la sesión
    /tokens N         tokens máximos por respuesta (por defecto 120)
    /temp X           temperatura de muestreo (por defecto 0.8; 0 = greedy)
    /crudo            alterna modo crudo (no envolver el prompt)

Uso:
    python examples/chat_code_llm.py
    python examples/chat_code_llm.py --ckpt benchmarks/data/code_llm.ckpt
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

import tensorc as tc
from code_llm import CKPT, CONFIG_JSON, TOK_JSON, CodeGPT

import json


def load_model(ckpt_path):
    from tokenizers import Tokenizer
    with open(CONFIG_JSON, encoding="utf-8") as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(TOK_JSON)
    eos_id = tok.token_to_id("<|endoftext|>")
    model = CodeGPT(cfg["vocab"], cfg["dim"], cfg["ff"], cfg["heads"],
                    cfg["layers"], cfg["seq"])
    saved = tc.load(ckpt_path, requires_grad=True)
    if isinstance(saved, tc.Tensor):
        saved = [saved]
    assert len(saved) == len(model.params), \
        "checkpoint incompatible con la configuración"
    for p, s in zip(model.params, saved):
        p.copy_(s)
    return model, tok, eos_id, cfg["seq"]


def generate(model, tok, eos_id, seq, prompt, max_tokens, temperature):
    """Genera token a token; devuelve (texto_nuevo, tokens/seg)."""
    ids = tok.encode(prompt).ids[-seq:]
    rng = np.random.default_rng()
    out_ids = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        ctx = ([eos_id] * (seq - len(ids)) + ids)[-seq:]
        xb = np.asarray([ctx], dtype=np.float32)
        probs = model.forward(tc.Tensor(xb))
        last = probs.slice_axis(0, seq - 1, 1)
        p = np.asarray(last.data, dtype=np.float64).reshape(-1)
        if temperature <= 0.0:
            nxt = int(p.argmax())
        else:
            p = np.log(p + 1e-12) / temperature
            p = np.exp(p - p.max())
            nxt = int(rng.choice(model.vocab, p=p / p.sum()))
        if nxt == eos_id:
            break
        ids.append(nxt)
        out_ids.append(nxt)
    dt = time.perf_counter() - t0
    rate = len(out_ids) / dt if dt > 0 else 0.0
    return tok.decode(out_ids), rate


def wrap_prompt(text):
    """El modelo es de continuación de código: un comentario estilo
    docstring/comentario ayuda a que responda con código."""
    stripped = text.strip()
    code_cues = ("def ", "class ", "import ", "from ", "#!", "#", "if ",
                 "for ", "while ", "$", "echo ", "cd ", "sudo ")
    if stripped.startswith(code_cues):
        return stripped
    return "# %s\ndef " % stripped


HELP = """comandos: /salir, /tokens N, /temp X (0=greedy), /crudo (prompt tal cual)
Escribe código o una petición en lenguaje natural (se envuelve como
comentario). Ten paciencia: ~0.5 s por token en esta GPU bajo WSL2."""


def main():
    ap = argparse.ArgumentParser(description="Chat de terminal con el code-LLM")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--crudo", action="store_true")
    args = ap.parse_args()

    print("cargando modelo desde %s ..." % args.ckpt)
    model, tok, eos_id, seq = load_model(args.ckpt)
    print("listo (%s, seq %d). %s\n" % (args.ckpt, seq, HELP))

    max_tokens, temperature, crudo = args.tokens, args.temp, args.crudo
    while True:
        try:
            text = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nadiós")
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, val = text.partition(" ")
            if cmd == "/salir":
                print("adiós")
                break
            if cmd == "/tokens" and val.isdigit():
                max_tokens = int(val)
                print("  tokens por respuesta: %d" % max_tokens)
            elif cmd == "/temp":
                try:
                    temperature = float(val)
                    print("  temperatura: %.2f" % temperature)
                except ValueError:
                    print("  uso: /temp 0.8")
            elif cmd == "/crudo":
                crudo = not crudo
                print("  modo crudo: %s" % ("sí" if crudo else "no"))
            else:
                print(HELP)
            continue
        prompt = text if crudo else wrap_prompt(text)
        reply, rate = generate(model, tok, eos_id, seq, prompt,
                               max_tokens, temperature)
        print("modelo>%s" % reply)
        print("  [%.1f tokens/s]" % rate)


if __name__ == "__main__":
    main()
