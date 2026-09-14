#!/bin/bash

commands=(
    # "python train_oppu.py --task_name news_headline --output_dir ./output/news_headline/OPPU/LoRA --lora_ckpt ./output/news_headline/task-base_LLM/lora_ckpt.pt"
    # "python train_oppu.py --task_name scholarly_title --output_dir ./output/scholarly_title/OPPU/LoRA --lora_ckpt ./output/scholarly_title/task-base_LLM/lora_ckpt.pt"
    # "python train_oppu.py --task_name product_rating --output_dir ./output/product_rating/OPPU/LoRA --lora_ckpt ./output/product_rating/task-base_LLM/lora_ckpt.pt"
    
    "python oppu.py --task_name news_headline --output_dir ./output/news_headline/OPPU --lora_ckpt ./output/news_headline/task-base_LLM/lora_ckpt.pt --max_gen_len 100"
    "python oppu.py --task_name scholarly_title --output_dir ./output/scholarly_title/OPPU --lora_ckpt ./output/scholarly_title/task-base_LLM/lora_ckpt.pt --max_gen_len 100"
    # "python oppu.py --task_name product_rating --output_dir ./output/product_rating/OPPU --lora_ckpt ./output/product_rating/task-base_LLM/lora_ckpt.pt"
    
)

for cmd in "${commands[@]}"; do
    echo "========================================"
    echo "$cmd"
    eval "$cmd"
done

echo "All finished."