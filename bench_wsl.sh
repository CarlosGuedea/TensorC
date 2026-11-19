#!/bin/bash
# Reconstruye la extensión Python con CUDA y mide el LLM (misma carga que
# la medición anterior: 300 pasos, dim 64, 2 bloques).
set -e
export PATH="$HOME/tensorc-env/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/tensorc-env/lib:$LD_LIBRARY_PATH"
cd ~/tensorc-src/python
TENSORC_CUDA=1 CUDA_HOME="$HOME/tensorc-env" python3 setup.py build_ext --inplace 2>&1 | grep -E "error|Error" || echo "extensión OK"
cd ~/tensorc-src
python3 -c "
import sys, time
sys.path.insert(0, 'python')
import tensorc
print('cuda_available:', tensorc.cuda_available())
"
echo "=== LLM 300 pasos (GPU) ==="
python3 - <<'EOF'
import sys, time, subprocess
sys.path.insert(0, 'python')
# Corre el demo como script para reutilizar su loop exacto
import runpy
t0 = time.perf_counter()
sys.argv = ['char_llm_demo.py', '300']
runpy.run_path('examples/char_llm_demo.py', run_name='__main__')
EOF
