#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

LLAMA_FACTORY_ROOT="/home/ubuntu/QiangYi/LLaMA-Factory"
DATASET_DIR_ABS="$LLAMA_FACTORY_ROOT/data"
MEDIA_DIR_ABS="/home/ubuntu/HFRL_fix"
OUTPUT_DIR_ABS="$LLAMA_FACTORY_ROOT/saves/qwen2_vl/lora_expanded/sft"

cat > ds_z3_config.json << 'EOF'
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
EOF

echo "开始8卡纯训练模式（完全禁用评估）..."
echo "模型: /home/ubuntu/qwen2-vl"
echo "数据集: hfrl_data_expanded_qwen"
echo "数据集目录 (绝对): $DATASET_DIR_ABS"
echo "媒体文件目录 (绝对): $MEDIA_DIR_ABS"
echo "输出目录 (绝对): $OUTPUT_DIR_ABS"
echo "---"

FORCE_TORCHRUN=1 llamafactory-cli train \
    --deepspeed ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/ubuntu/qwen2-vl \
    --dataset hfrl_data_expanded_qwen \
    --dataset_dir "$DATASET_DIR_ABS" \
    --media_dir "$MEDIA_DIR_ABS" \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir "$OUTPUT_DIR_ABS" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 10000 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --evaluation_strategy "no" \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.00 \
    --flash_attn auto \
    --bf16 true \
    --ddp_find_unused_parameters false

EXIT_CODE=$?
echo "训练结束，退出码: $EXIT_CODE"
exit $EXIT_CODE
