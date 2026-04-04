#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

missing=0

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    echo "[ok]   $label -> $path"
  else
    echo "[miss] $label -> $path"
    missing=1
  fi
}

check_path "$REPO_ROOT/external/RNA-FM" "RNA-FM repository"
check_path "$REPO_ROOT/external/RNA-FM/RNA-FM_pretrained.pth" "RNA-FM checkpoint"

check_path "$REPO_ROOT/external/ERNIE-RNA" "ERNIE-RNA repository"
check_path "$REPO_ROOT/external/ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt" "ERNIE-RNA checkpoint"
check_path "$REPO_ROOT/external/ERNIE-RNA/src/dict" "ERNIE-RNA token dictionary"

check_path "$REPO_ROOT/external/RNABERT" "RNABERT repository"
check_path "$REPO_ROOT/external/RNABERT/MLM_SFP.py" "RNABERT entry script"
check_path "$REPO_ROOT/external/RNABERT/bert_mul_2.pth" "RNABERT checkpoint"
check_path "$REPO_ROOT/external/RNABERT/RNA_bert_config.json" "RNABERT config"

check_path "$REPO_ROOT/external/RiNALMo" "RiNALMo repository"
check_path "$REPO_ROOT/external/RiNALMo/weights/rinalmo_giga_pretrained.pt" "RiNALMo checkpoint"

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "Some external assets are missing."
  echo "See external/README.md and external/model_sources.lock.json for acquisition notes."
  exit 1
fi

echo
echo "All required external assets are present."

