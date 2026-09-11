#!/usr/bin/env bash
set -euo pipefail

# Download DINOv3 ViT-L/16 distilled and ViT-H+/16 distilled weights.
#
# These models are gated. Before running, log in in a browser and accept access:
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#   https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m
#
# Run one of:
#   MODE=hf_safetensors HF_TOKEN=hf_xxx bash scripts/download_dinov3_weights.sh
#   MODE=meta_pth DINO_VITL16_PTH_URL='https://...' DINO_VITH16PLUS_PTH_URL='https://...' bash scripts/download_dinov3_weights.sh
#
# Output for MODE=hf_safetensors:
#   /mnt/data_16t/football/qiuqi/checkpoints/dinov3_vitl16_pretrain_lvd1689m/model.safetensors
#   /mnt/data_16t/football/qiuqi/checkpoints/dinov3_vith16plus_pretrain_lvd1689m/model.safetensors
#
# Output for MODE=meta_pth:
#   /mnt/data_16t/football/qiuqi/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
#   /mnt/data_16t/football/qiuqi/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
#
# The train_football_events.py script in this repo expects the Meta .pth files.

OUT_DIR="${OUT_DIR:-/mnt/data_16t/football/qiuqi/checkpoints}"
MODE="${MODE:-hf_safetensors}"
HF_TOKEN="${HF_TOKEN:-}"
DINO_VITL16_PTH_URL="${DINO_VITL16_PTH_URL:-}"
DINO_VITH16PLUS_PTH_URL="${DINO_VITH16PLUS_PTH_URL:-}"

mkdir -p "${OUT_DIR}"

download_file() {
  local url="$1"
  local out_path="$2"
  local auth_header="${3:-}"

  mkdir -p "$(dirname "${out_path}")"
  echo "Downloading ${url}"
  echo "  -> ${out_path}"

  if command -v curl >/dev/null 2>&1; then
    local curl_args=(
      --fail
      --location
      --continue-at -
      --retry 5
      --retry-delay 5
    )
    if [[ -n "${auth_header}" ]]; then
      curl_args+=(--header "${auth_header}")
    fi
    curl "${curl_args[@]}" "${url}" --output "${out_path}"
  elif command -v wget >/dev/null 2>&1; then
    local wget_args=(
      --continue
      --output-document="${out_path}"
    )
    if [[ -n "${auth_header}" ]]; then
      wget_args+=(--header="${auth_header}")
    fi
    wget "${wget_args[@]}" "${url}"
  else
    echo "ERROR: neither curl nor wget is available."
    exit 1
  fi

  echo "Done: ${out_path}"
}

if [[ "${MODE}" == "hf_safetensors" ]]; then
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Run: MODE=hf_safetensors HF_TOKEN=hf_xxx bash scripts/download_dinov3_weights.sh"
    exit 1
  fi

  download_file \
    "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/resolve/main/model.safetensors" \
    "${OUT_DIR}/dinov3_vitl16_pretrain_lvd1689m/model.safetensors" \
    "Authorization: Bearer ${HF_TOKEN}"

  download_file \
    "https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/main/model.safetensors" \
    "${OUT_DIR}/dinov3_vith16plus_pretrain_lvd1689m/model.safetensors" \
    "Authorization: Bearer ${HF_TOKEN}"

  echo
  echo "Downloaded Hugging Face safetensors under ${OUT_DIR}/"
  echo "These are best used through Hugging Face Transformers."
elif [[ "${MODE}" == "meta_pth" ]]; then
  if [[ -z "${DINO_VITL16_PTH_URL}" || -z "${DINO_VITH16PLUS_PTH_URL}" ]]; then
    echo "ERROR: DINO_VITL16_PTH_URL and DINO_VITH16PLUS_PTH_URL must be set."
    echo "Use the official Meta .pth URLs you receive after accepting DINOv3 access."
    exit 1
  fi

  download_file \
    "${DINO_VITL16_PTH_URL}" \
    "${OUT_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

  download_file \
    "${DINO_VITH16PLUS_PTH_URL}" \
    "${OUT_DIR}/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

  echo
  echo "Downloaded Meta .pth weights under ${OUT_DIR}/"
  echo "Use these with train_football_events.py:"
  echo "  model.weights=${OUT_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  echo "  model.weights=${OUT_DIR}/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
else
  echo "ERROR: unknown MODE=${MODE}"
  echo "Expected MODE=hf_safetensors or MODE=meta_pth"
  exit 1
fi
