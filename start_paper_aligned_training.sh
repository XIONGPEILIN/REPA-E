#!/bin/bash
export WANDB_DIR="/export/ssd2/xiong-p/wandb_repae"
export HF_HOME="/export/ssd2/xiong-p/hf_home"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE="0"
export NCCL_SHM_DISABLE="1"
export NCCL_P2P_LEVEL="5"
export CUDA_VISIBLE_DEVICES="4,5,6,7"

accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    train_repae.py \
    --use-hf-imagenet \
    --resolution=256 \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="bf16" \
    --seed=0 \
    --output-dir="exps" \
    --batch-size=256 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --checkpointing-steps=50000 \
    --loss-cfg-path="configs/l1_lpips_kl_gan.yaml" \
    --vae="f8d4" \
    --vae-ckpt="pretrained/sdvae/sdvae-f8d4.pt" \
    --disc-pretrained-ckpt="pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --exp-name="sit-xl-dinov2-b-repae" \
    --compile \
    --compile-mode="max-autotune-no-cudagraphs" \
    --resume-step=0 \
    --continue-train-exp-dir="outputs/sit-xl-dinov2-b-repae-student"
