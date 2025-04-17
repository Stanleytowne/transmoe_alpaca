#! /bin/bash
 
#SBATCH -J tpz
#SBATCH -p IAI_SLURM_HGX
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH -c 32

torchrun --nproc_per_node=4 --master_port=12345 train.py \
    --model_name_or_path ./output/alpacamoe-2-7B-onlyrouter \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./output/alpacamoe-2-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/ds_config_zero2_no_offload.json" \
    --tf32 True