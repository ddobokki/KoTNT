#!/bin/bash
GPU_IDS="0,1,2"

# 1640000
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 -m torch.distributed.launch \
	--nproc_per_node 3 ../bart_train.py \
	--datasets_dirs="" \
	--model_name_or_path="" \
	--output_dir="" \
	--overwrite_output_dir \
	--save_total_limit="3" \
	--max_steps="" \
	--save_strategy="steps" \
	--save_steps="" \
	--evaluation_strategy="steps" \
	--eval_steps="" \
	--logging_steps="" \
	--warmup_steps="" \
	--lr_scheduler_type="" \
	--optim="adamw_torch" \
	--adam_beta2="" \
	--learning_rate="" \
	--weight_decay="" \
	--per_device_train_batch_size="" \
    --gradient_accumulation_steps="" \
	--per_device_eval_batch_size="" \
	--seed="" \
    --cache_dir="./.cache" \
	--group_by_length \
	--fp16 \
	--metric_for_best_model="eval_loss" \
    --load_best_model_at_end \
	--do_train \
	--do_eval \
	--eval_size="0.1" \
	--setproctitle_name="" \
    --wandb_project="" \
    --wandb_entity="" \
    --wandb_name="" \
    --greater_is_better="false" \
	--predict_with_generate="false" \
	--direction="forward"