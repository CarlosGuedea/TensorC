"""Valida los logits de QwenTensorC contra la referencia de HuggingFace.

Corre el mismo prompt por ambos modelos (HF en fp32 CPU) y compara los
logits de todas las posiciones: diferencia absoluta máxima/media y top-5
de la última posición.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from qwen_tensorc import DATA_DIR, load_model

import tensorc as tc
from tokenizers import Tokenizer


def main():
    import torch
    from transformers import AutoModelForCausalLM

    prompt = sys.argv[1] if len(sys.argv) > 1 else "def quicksort(arr):"
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    ids = tok.encode(prompt).ids
    print(f"prompt: {prompt!r} ({len(ids)} tokens)")

    # Referencia HuggingFace en fp32.
    hf = AutoModelForCausalLM.from_pretrained(
        DATA_DIR, torch_dtype=torch.float32)
    hf.eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].numpy()

    # TensorC.
    model = load_model(max_seq=64)
    x = tc.Tensor(np.array([ids], dtype=np.float32))
    mine = np.array(model.forward(x).data, dtype=np.float32).reshape(
        len(ids), -1)

    diff = np.abs(mine - ref)
    print(f"max|diff|  = {diff.max():.4f}")
    print(f"mean|diff| = {diff.mean():.6f}")
    cos = float(np.dot(mine.ravel(), ref.ravel()) /
                (np.linalg.norm(mine) * np.linalg.norm(ref)))
    print(f"cosine     = {cos:.6f}")

    top_mine = np.argsort(mine[-1])[-5:][::-1]
    top_ref = np.argsort(ref[-1])[-5:][::-1]
    print("top-5 TensorC:", [(int(t), repr(tok.decode([int(t)])))
                             for t in top_mine])
    print("top-5 HF     :", [(int(t), repr(tok.decode([int(t)])))
                             for t in top_ref])


if __name__ == "__main__":
    main()
