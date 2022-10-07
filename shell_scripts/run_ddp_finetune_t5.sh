#!/bin/bash

script_path="/data/jsb193/github/t5/T5/main.py"

model_name_or_path="/data/jsb193/github/t5/output/txt_to_num/s/checkpoint-10000"
checkpoint_path="/data/jsb193/github/t5/output/text_to_num/checkpoint-9000"
data_name_or_script="/data/jsb193/github/t5/data/post_out.csv"
output_dir="/data/jsb193/github/t5/output/num_to_txt/m"
cache_dir="/data/jsb193/github/t5/.cache"

export CUDA_VISIBLE_DEVICES="0,2,3"
export WANDB_DISABLED="false"
export WANDB_PROJECT="konuko-T5"
export WANDB_ENTITY="tadev"
export WANDB_CACHE_DIR=$cache_dir
export WANDB_USERNAME="jp_42maru"
export WANDB_RUN_GROUP="finetune"
export WANDB_TAGS="T5, finetune, multi-finetune"
export WANDB_DISABLE_CODE="false"
# export WANDB_RESUME="allow"
# export WANDB_RUN_ID="zilo0iv2"
export OMP_NUM_THREADS=8
# --resume_from_checkpoint=$checkpoint_path \

python -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=3 \
    $script_path \
    --run_name="[JP]num-to-txt T5 m" \
    --model_name=$model_name_or_path \
    --data_name=$data_name_or_script \
    --output_dir=$output_dir \
    --cache=$cache_dir \
    --per_device_train_batch_size=12 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --eval_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --warmup_steps=1000 \
    --num_train_epochs=2 \
    --lr_scheduler_type="linear" \
    --logging_strategy="steps" \
    --logging_steps=50 \
    --evaluation_strategy="steps" \
    --eval_steps=1000 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --do_train \
    --do_eval \
    --group_by_length \
    --fp16\
    --num_proc=10