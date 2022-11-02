#!/bin/bash
GPU_IDS="0,1,2,3"
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python -m torch.distributed.launch --nproc_per_node 4 train.py \
    --train_type "NTT" \
	--train_csv_paths "data/csv/train.csv" \
	--valid_csv_paths "data/csv/valid.csv" \
	--model_name_or_path "skt/kogpt2-base-v2" \
	--output_dir "output/" \
	--overwrite_output_dir \
	--save_total_limit "2" \
	--num_train_epochs "1" \
	--save_strategy "steps" \
	--evaluation_strategy "steps" \
	--logging_steps "500" \
	--save_steps "500" \
	--eval_steps "500" \
	--optim "adamw_torch" \
	--learning_rate "2e-5" \
	--per_device_train_batch_size "8" \
	--per_device_eval_batch_size "8" \
    --gradient_accumulation_steps "1" \
	--seed "42" \
    --cache_dir ".cache" \
    --load_best_model_at_end \
	--do_train \
	--do_eval \
	--setproctitle_name "num_converter" \
    --wandb_project="TNT" \
    --wandb_entity "ddobokki" \
    --wandb_name "test" \
	--dataloader_num_workers "4" \
	--num_proc "8" \
    --greater_is_better "false" \
	--predict_with_generate "false" \
	--metric_for_best_model "eval_loss" \
	--fp16 