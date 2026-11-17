#!/usr/bin/env python3
"""Comparación reproducible Conv2D: TensorC (C11) contra una implementación
equivalente en Python/NumPy, con exactamente los mismos datos y dimensiones.

El script:
  1. Genera entrada y kernel float32 deterministas (semilla fija) y los
     guarda en archivos binarios row-major.
  2. Ejecuta la convolución con el mismo algoritmo directo en NumPy y la
     temporiza en milisegundos (mediana y mínimo).
  3. Ejecuta el binario benchmark de TensorC sobre los MISMOS archivos y
     temporiza el forward y el backward en C.
  4. Verifica que la salida de C coincide con NumPy y reporta la tabla.

Uso:
  python3 conv2d_numpy.py [--c-bin RUTA] [--reps-python N] [--reps-c N]
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Dimensiones del benchmark: [N, C, H, W] x [K, C, KH, KW], stride, padding.
N, C, H, W = 8, 3, 32, 32
K, KH, KW = 16, 3, 3
SH, SW, PH, PW = 1, 1, 1, 1
SEED = 42


def conv2d_numpy(x, k, sh, sw, ph, pw):
    """Cross-correlación 2D directa, equivalente al algoritmo en C."""
    xpad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    n, _, hp, wp = xpad.shape
    kk, _, kh, kw = k.shape
    oh = (hp - kh) // sh + 1
    ow = (wp - kw) // sw + 1
    out = np.empty((n, kk, oh, ow), dtype=np.float32)
    for b in range(n):
        for f in range(kk):
            for i in range(oh):
                for j in range(ow):
                    window = xpad[b, :, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    out[b, f, i, j] = np.sum(window * k[f], dtype=np.float32)
    return out


def summarize(samples):
    samples = np.asarray(samples, dtype=np.float64)
    return float(np.median(samples)), float(samples.min())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--c-bin",
                        default=str((here / "../../build/tensorc_conv2d_benchmark")
                                    .resolve()))
    parser.add_argument("--reps-python", type=int, default=7)
    parser.add_argument("--reps-c", type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N, C, H, W), dtype=np.float32)
    k = rng.standard_normal((K, C, KH, KW), dtype=np.float32)

    tmp = Path(tempfile.mkdtemp(prefix="tensorc_conv2d_"))
    x_path = tmp / "input.bin"
    k_path = tmp / "kernel.bin"
    x.tofile(x_path)
    k.tofile(k_path)

    # --- NumPy: calentamiento + medición ---
    numpy_out = conv2d_numpy(x, k, SH, SW, PH, PW)
    numpy_times = []
    for _ in range(args.reps_python):
        start = time.perf_counter()
        out = conv2d_numpy(x, k, SH, SW, PH, PW)
        numpy_times.append((time.perf_counter() - start) * 1000.0)
        assert np.array_equal(out, numpy_out), "NumPy no es determinista"
    numpy_med, numpy_min = summarize(numpy_times)

    # --- TensorC: el binario lee los mismos archivos binarios ---
    cmd = [args.c_bin, str(x_path), str(k_path),
           str(N), str(C), str(H), str(W),
           str(K), str(C), str(KH), str(KW),
           str(SH), str(SW), str(PH), str(PW),
           str(args.reps_c)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(2)
    metrics = {}
    for line in proc.stdout.splitlines():
        key, _, value = line.partition(" ")
        metrics[key] = float(value)

    # --- Exactitud: la salida de C debe coincidir con NumPy ---
    c_out_path = tmp / "c_output.bin"
    max_diff = float("nan")
    c_verified = "n/d (el binario no volcó c_output.bin)"
    if c_out_path.exists():
        c_out = np.fromfile(c_out_path, dtype=np.float32).reshape(numpy_out.shape)
        max_diff = float(np.max(np.abs(c_out.astype(np.float64) -
                                       numpy_out.astype(np.float64))))
        c_verified = "sí" if max_diff < 1e-3 else "NO"

    print(f"Dimensiones: input [{N},{C},{H},{W}], kernel [{K},{C},{KH},{KW}], "
          f"stride ({SH},{SW}), padding ({PH},{PW})")
    print(f"Repeticiones: NumPy {args.reps_python}, C {args.reps_c}")
    print()
    print(f"{'implementación':<16}{'forward mediana (ms)':>24}{'forward mín (ms)':>20}"
          f"{'backward mediana (ms)':>26}")
    print(f"{'NumPy':<16}{numpy_med:>24.4f}{numpy_min:>20.4f}{'n/d':>26}")
    print(f"{'TensorC (C11)':<16}"
          f"{metrics['FORWARD_MS_MEDIAN']:>24.4f}"
          f"{metrics['FORWARD_MS_MIN']:>20.4f}"
          f"{metrics['BACKWARD_MS_MEDIAN']:>26.4f}")
    print()
    ratio = numpy_med / metrics["FORWARD_MS_MEDIAN"]
    print(f"TensorC/NumPy (forward, mediana): {ratio:.2f}x")
    print(f"Salida C == NumPy (máx. |diff| = {max_diff:.3e}): {c_verified}")


if __name__ == "__main__":
    main()
