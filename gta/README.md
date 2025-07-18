### 🚀 Training Details
To reproduce the training curve and performance, you can use the `run_clm.py` provided by HuggingFace. The exact training hyperparameter are as follows:

```python

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --rdzv_endpoint $ADDR:$PORT \
    --rdzv_conf=join_timeout=36000000,read_timeout=3600000,timeout=36000000 \
    "

clm_options=" \
    --train_file $DATA \
    --trust_remote_code true \
    --attn $ATTN \
    --experiment_id $DATE \
    --report_to wandb \
    --block_size $BLOCK_SIZE \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 10 \
    --learning_rate $LR \
    --logging_steps 1 \
    --num_train_epochs $EPOCH \
    --bf16 true \
    --config_name $CONFIG \
    --tokenizer_name $CONFIG \
    --model_type $MODEL_TYPE \
    --per_device_train_batch_size $MICRO_BATCH \
    --gradient_accumulation_steps $BATCH_ACC \
    --optim adamw_hf \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARM_RATIO \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps $SAVE_STEP \
    --deepspeed $DEEPSPEED \
    --overwrite_output_dir \
    --output_dir $SAVED_PRETRAIN_CHECKPOINT_PATH \
    --cache_dir $CACHE \
    --max_eval_samples 100 \
    --do_train \

SCRIPTS="run_clm_run.py"
run_cmd="torchrun $DISTRIBUTED_ARGS $SCRIPTS
 ${clm_options} ${eval_options}"

echo ${run_cmd}
eval ${run_cmd}

```
