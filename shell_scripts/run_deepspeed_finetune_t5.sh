script_path=""

model_name_or_path=""
checkpoint_path=""
data_name_or_script=""
output_dir=""
cache_dir=""

gpu_num = 1

export CUDA_VISIBLE_DEVICES=""
export WANDB_DISABLED=""
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_CACHE_DIR=$cache_dir
export WANDB_USERNAME=""
export WANDB_RUN_GROUP=""
export WANDB_TAGS=""
export WANDB_DISABLE_CODE=""
# export WANDB_RESUME=""
# export WANDB_RUN_ID=""
export OMP_NUM_THREADS=8

# --resume_from_checkpoint=$checkpoint_path \
deepspeed --num_gpu $gpu_num script_path \
    --run_name="" \
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
    --fp16 \
    --num_proc=10