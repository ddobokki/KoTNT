#!/bin/bash
GPU_IDS="0,1,2"

# 1640000
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 -m torch.distributed.launch \
	--nproc_per_node 3 ../bart_train.py \
	--datasets_dirs="/data2/bart/temp_workspace/nlp/KoGPT_num_converter/data/csv/post_out.csv" \
	--model_name_or_path="/data2/bart/temp_workspace/nlp/models/kobart-base-v2" \
	--output_dir="/data2/bart/temp_workspace/nlp/output_dir" \
	--overwrite_output_dir \
	--save_total_limit="3" \
	--max_steps="650000" \
	--save_strategy="steps" \
	--save_steps="10000" \
	--evaluation_strategy="steps" \
	--eval_steps="10000" \
	--logging_steps="2000" \
	--warmup_steps="70000" \
	--lr_scheduler_type="cosine" \
	--optim="adamw_torch" \
	--learning_rate="2e-5" \
	--weight_decay="0" \
	--per_device_train_batch_size="2" \
    --gradient_accumulation_steps="1" \
	--per_device_eval_batch_size="1" \
	--seed="42" \
    --cache_dir="./.cache" \
	--group_by_length \
	--fp16 \
	--metric_for_best_model="eval_loss" \
    --load_best_model_at_end \
	--do_train \
	--do_eval \
	--eval_size="0.1" \
	--setproctitle_name="bart" \
    --wandb_project="BartForConditionalGeneration" \
    --wandb_entity="bart_tadev" \
    --wandb_name="num_to_ko" \
    --greater_is_better="false" \
	--predict_with_generate="false"