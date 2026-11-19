#!/bin/bash
# Recompila y corre el benchmark de matmul C contra la v3.
set -e
export PATH="$HOME/tensorc-env/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tensorc-env/lib:$LD_LIBRARY_PATH"
cd ~/tensorc-src
gcc -O2 -I include nd_matmul_bench.c build-cuda/libtensorc.a \
    -L "$HOME/tensorc-env/lib" -lcudart -lcublas -lm -o nd_matmul_bench
./nd_matmul_bench
