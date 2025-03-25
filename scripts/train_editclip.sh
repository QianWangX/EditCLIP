# Single-GPU training:

python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --dataset-type ip2p \
    --warmup 1 \
    --batch-size=256 \
    --lr=2e-6 \
    --lr-newconv=2e-4 \
    --wd=2e-2 \
    --epochs=40 \
    --workers=6 \
    --model "ViT-L-14" \
    --pretrained "openai" \
    --precision amp \
    --log-every-n-steps 5 \
    --save-ckpt-every-n-steps 50000

# Multi-GPU training:

cd open_clip/src
torchrun --nproc_per_node 2 -m open_clip_train.main \
--save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to wandb \
    --dataset-type ip2p \
    --warmup 1 \
    --batch-size=256 \
    --lr=2e-6 \
    --lr-newconv=2e-4 \
    --wd=2e-2 \
    --epochs=40 \
    --workers=6 \
    --model "ViT-L-14" \
    --pretrained "openai" \
    --precision amp \
    --log-every-n-steps 5 \
    --save-ckpt-every-n-steps 50000