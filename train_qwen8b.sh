#!/bin/bash
set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export REPO=/workspace/Empirical-Influence-Function-main
export MODEL=/workspace/models/Qwen3-8B
export DATA=/workspace/data/annotated/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl
export OUT=$REPO/outputs/qwen3-8b-ce-saliency-raw-10k

export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=7200
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

cd "$REPO"

torchrun --standalone --nproc_per_node=7 src/train/train.py \
  --model_name_or_path "$MODEL" \
  --data_path "$DATA" \
  --output_dir "$OUT" \
  --use_peft True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --loss_mode ce_saliency \
  --saliency_loss_type contrastive \
  --saliency_lambda 1.5 \
  --saliency_alpha 1.0 \
  --saliency_margin_plus 2.0 \
  --saliency_layer -1 \
  --saliency_neg_sample_k 64 \
  --num_train_epochs 20 \
  --gradient_checkpointing True \
  --torch_empty_cache_steps 20 \
  --ddp_find_unused_parameters False \
  --enable_attn_viz False \
  --saliency_detail_log_steps 0 \
  --eval_codebleu_samples 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --ddp_find_unused_parameters False \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --max_len 1024 \
  --bf16 True \
  --use_flash_attention False \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --dataloader_num_workers 2 \
  --report_to none \
  --run_name qwen3-8b-ce-saliency-raw-10k \
  --remove_unused_columns False \
  --supervise_eos False \