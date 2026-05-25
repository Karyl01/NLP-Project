#!/usr/bin/env bash
# 下载 Qwen2.5-1.5B-Instruct（支持 HF 镜像 / ModelScope，避免直连超时）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

METHOD="${1:-auto}"
echo "项目目录: ${ROOT}"
echo "下载方式: ${METHOD} (可选: auto | hf_mirror | modelscope)"

python scripts/download_baseline_model.py --method "${METHOD}"
