peft=lora

for ratio in $(seq 0.10 0.10 0.90)
do
CUDA_VISIBLE_DEVICES=0, python run_gpt2_pos_traintime_truncation.py \
    --model_name_or_path togethercomputer/RedPajama-INCITE-Base-3B-v1 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 3 \
    --block_size 256 \
    --preprocessing_num_workers 12 \
    --output_dir ./output/output_${peft}_ratio${ratio}_redpajama_localattn \
    --compress \
    --bound_ratio $ratio \
    --peft $peft \
    --r 16
done