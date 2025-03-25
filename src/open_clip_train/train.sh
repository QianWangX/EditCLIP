srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main_transfer.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --dataset-type ip2p \
    --warmup 1 \
    --batch-size=256 \
    --lr=2e-6 \
    --lr-newconv=2e-4 \
    --wd=2e-2 \
    --epochs=50 \
    --workers=6 \
    --model "ViT-L-14" \
    --pretrained "openai" \
    --precision amp \
    --log-every-n-steps 5 \