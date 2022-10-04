#!/bin/bash

script_path=""

model_name_or_path=""
data_name_or_script=""
output_dir=""
cache_dir=""

# T5 finetune script
export OMP_NUM_THREADS=8
python $script_path \
    --model_name_or_path=$model_name_or_path \
    --data_name_or_script=$data_name_or_script \
    --output_dir=$output_dir \
    --cache=$cache_dir \
    --per_device_train_batch_size = 10 \
    --per_device_eval_batch_size = 4 \
    --gradient_accumulation_steps = 2 \
    --eval_accumulation_steps = 2 \
    --learning_rate = 1e-5 \
    --warmup_steps = 1000 \
    --num_train_epochs = 2 \
    --lr_scheduler_type = "linear" \
    --logging_strategy = "steps" \
    --logging_steps = 1 \
    --evaluation_strategy = "steps" \
    --eval_steps = 5000 \
    --do_train true \
    --num_proc = 6