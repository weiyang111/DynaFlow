#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
mkdir -p "$RAW_DIR"

fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out already exists"
    return
  fi
  echo "[download] $url -> $out"
  curl -L "$url" -o "$out"
}

fetch "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz" "$RAW_DIR/bitcoin_alpha.csv.gz"
fetch "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz" "$RAW_DIR/bitcoin_otc.csv.gz"
fetch "https://snap.stanford.edu/data/CollegeMsg.txt.gz" "$RAW_DIR/uci_messages.txt.gz"
fetch "https://nrvis.com/download/data/dynamic/ia-digg-reply.zip" "$RAW_DIR/digg.zip"
fetch "https://nrvis.com/download/data/dynamic/email-dnc.zip" "$RAW_DIR/email_dnc.zip"
fetch "https://nrvis.com/download/data/dynamic/tech-as-topology.zip" "$RAW_DIR/topology.zip"

echo "All downloads completed under $RAW_DIR"
