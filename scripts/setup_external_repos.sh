#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"

mkdir -p "$EXTERNAL_DIR"

clone_or_update() {
  local name="$1"
  local url="$2"
  local branch="$3"
  local commit="$4"
  local dest="$EXTERNAL_DIR/$name"

  if [[ ! -d "$dest/.git" ]]; then
    echo "[clone] $name"
    git clone --branch "$branch" "$url" "$dest"
  else
    echo "[fetch] $name"
    git -C "$dest" fetch origin
  fi

  echo "[checkout] $name -> $commit"
  git -C "$dest" checkout "$commit"
}

clone_or_update "RNA-FM" "https://github.com/ml4bio/RNA-FM.git" "main" "348951516e0963d22bbb33b3c9fc18c89081d38e"
clone_or_update "ERNIE-RNA" "https://github.com/Bruce-ywj/ERNIE-RNA.git" "main" "43bc06de1088ed03ffd7de918ad4b2c2a3346a43"
clone_or_update "RNABERT" "https://github.com/mana438/RNABERT.git" "master" "1aeebcb2823bc34fc37f6527d63fca06917e3919"
clone_or_update "RiNALMo" "https://github.com/lbcb-sci/RiNALMo.git" "main" "2c2c5c14a5ae609d8c560a5d9ca32e51e0288955"

mkdir -p "$EXTERNAL_DIR/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint"
mkdir -p "$EXTERNAL_DIR/RiNALMo/weights"

echo
echo "Repositories are in place under $EXTERNAL_DIR"
echo "Next step: add any missing checkpoints, then run:"
echo "  bash scripts/check_external_assets.sh"

