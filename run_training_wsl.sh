#!/bin/bash
# Lanza una corrida de entrenamiento de code_llm en segundo plano.
# Uso: bash run_training_wsl.sh <log> [args de code_llm.py...]
set -e
LOG="$1"
shift
export LD_LIBRARY_PATH=/home/guede/tensorc-env/lib
cd /mnt/c/Users/guede/OneDrive/Documents/Proyectos/IA/TensorC-CUDA
setsid nohup ~/tensorc-env/bin/python -u examples/code_llm.py "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "pid=$PID"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "el proceso murió al arrancar; log:"
  cat "$LOG"
  exit 1
fi
head -5 "$LOG" || true
