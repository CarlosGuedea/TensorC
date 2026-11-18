#!/bin/bash
# Descarga el corpus de shell scripts desde repos públicos de GitHub y lo
# concatena en benchmarks/data/shell.txt (trozos de código Bash/Zsh reales).
set -e
TMP=$(mktemp -d)
cd "$TMP"
for repo in \
  https://github.com/pi-hole/pi-hole \
  https://github.com/RetroPie/RetroPie-Setup \
  https://github.com/Bash-it/bash-it \
  https://github.com/mathiasbynens/dotfiles \
  https://github.com/Nyr/openvpn-install \
  https://github.com/k4m4/movies-for-hackers \
  https://github.com/pyenv/pyenv \
  https://github.com/nvm-sh/nvm \
  https://github.com/rbenv/rbenv \
  https://github.com/ohmyzsh/ohmyzsh \
  https://github.com/junegunn/fzf \
  https://github.com/sorin-ionescu/prezto \
  https://github.com/VundleVim/Vundle.vim \
  https://github.com/tmux-plugins/tpm \
  https://github.com/agnoster/agnoster-zsh-theme \
  https://github.com/denysdovhan/spaceship-prompt ; do
  git clone --depth 1 --quiet "$repo" 2>/dev/null || echo "falló: $repo"
done
OUT=/mnt/c/Users/guede/OneDrive/Documents/Proyectos/IA/TensorC-CUDA/benchmarks/data/shell.txt
find . -type f \( -name '*.sh' -o -name '*.bash' -o -name '*.zsh' \) -size -200k -print0 | xargs -0 cat > "$OUT" 2>/dev/null || true
wc -c "$OUT"
rm -rf "$TMP"
