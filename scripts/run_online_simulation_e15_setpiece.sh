#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e14_signed_causal_continue4.yaml"
e14_dir="${repo_dir}/outputs/football_events/vitl16_online_simulation_e14_signed_causal_localgrad_from_last6r8_e1"
init_checkpoint="${E15_INIT_CHECKPOINT:-${e14_dir}/best.pt}"
output_dir="${E15_OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_online_simulation_e15_setpiece_subtype_context_from_e14_best}"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
subtype_weight="${E15_SUBTYPE_WEIGHT:-0.10}"
context_weight="${E15_CONTEXT_WEIGHT:-0.05}"

if [[ ! -f "${init_checkpoint}" ]]; then
  echo "Missing E15 init checkpoint: ${init_checkpoint}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${output_dir}" \
  "model.init_checkpoint=${init_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "train.epochs=4" \
  "train.scheduler_epochs=4" \
  "train.resume.enabled=false" \
  "train.set_piece_subtype_loss_weight=${subtype_weight}" \
  "train.set_piece_subtype_balance=masked_bce" \
  "train.set_piece_subtype_pos_weight.corner=1.12" \
  "train.set_piece_subtype_pos_weight.freekick=1.28" \
  "train.set_piece_subtype_pos_weight.penalty=12.0" \
  "train.set_piece_subtype_pos_weight.kickoff=8.0" \
  "train.raw_context_span_loss_weight=${context_weight}" \
  "train.raw_context_span_objective=mean_probability_floor" \
  "train.raw_context_span_min_probability=0.25" \
  "train.raw_context_span_softness=0.05" \
  2>&1 | tee -a "${output_dir}/train_console.log"

if [[ "${RUN_ONLINE_VAL_AFTER_TRAIN:-1}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_list}" \
    ONLINE_EVAL_WORLD_SIZE=4 \
    ONLINE_EVAL_PER_GPU_BATCH=4 \
    bash "${repo_dir}/scripts/evaluate_online_val15_checkpoint.sh" \
      "${base_config}" "${output_dir}/best.pt" \
      "${output_dir}/online_val15_best.json"
fi
