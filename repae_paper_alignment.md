# REPA-E 训练配置与官方论文对齐分析

## 关于 L1 与论文中 MSE 损失的差异说明

在论文（arXiv:2504.10483v3）中，作者写道：
> $\mathcal{L}_\mathrm{REG} = \mathcal{L}_\mathrm{KL} + \mathcal{L}_\mathrm{MSE} + \mathcal{L}_\mathrm{LPIPS} + \mathcal{L}_\mathrm{GAN}$

但在作者官方开源的代码中，配置 `configs/l1_lpips_kl_gan.yaml` 却明确使用的是 `reconstruction_loss: "l1"`。

**结论：作者开源的 L1 配置才是实际正确且效果更好的版本，论文中写 MSE 属于该领域的“表达惯例”（或笔误）。我们应该保持使用 L1。**

### 为什么会出现这种“文码不一”？
1. **生成质量的考量：** 在图像生成领域（特别是 VAE、VQGAN、Stable Diffusion系列底层），L2 损失（MSE）会过度惩罚大的像素级误差，且在误差较小时梯度极小，容易导致模型输出的图像变得**模糊（blurry）和过度平滑**。相反，L1 损失（MAE）对异常值更鲁棒，能够更好地保持图像的高频细节和锐利边缘。
2. **历史代码库生态继承：** 现代 Diffusion 模型的 VAE 大多沿用了 `taming-transformers` (VQGAN) 或后来的 `Stable Diffusion` (LDM) 代码库。在那些原始的经典开源库中，底层感知重建损失（`VQLPIPSWithDiscriminator`）默认都会采用 `l1_loss`。
3. **学术写作惯例：** 许多研究者在写论文时，习惯性地将“像素级重建损失”（Pixel-wise Reconstruction Loss）统称为 MSE 或 $\mathcal{L}_{rec}$，因为早期的自编码器（Autoencoder）大多是用 MSE 推导极大似然估计的。但他们在实际调参并发布代码时，为了最终的生成指标（FID、IS），始终都在用 L1。

### 修正决定
我们恢复使用官方提供的 `configs/l1_lpips_kl_gan.yaml`，不盲目将其强改为 L2。

## 最终对齐的启动配置
我们更新了启动脚本 `start_paper_aligned_training.sh`，采用如下严格修正项（包括硬件环境安全隔离与彻底的编译加速）：

```bash
#!/bin/bash
export WANDB_DIR="/export/ssd2/xiong-p/wandb_repae"
export HF_HOME="/export/ssd2/xiong-p/hf_home"

# 修正：GPU与分布式变量必须在外部 SH 声明，防止 accelerate 多进程派生错乱
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export NCCL_P2P_DISABLE="1"
export NCCL_SHM_DISABLE="1"

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=4 \
  train_repae.py \
  --resolution 256 \
  --max-train-steps 400000 \
  --checkpointing-steps 10000 \
  --use-hf-imagenet \
  --output-dir exps \
  --exp-name sit-xl-dinov2-b-repae-l1 \
  --allow-tf32 \
  --mixed-precision bf16 \
  --loss-cfg-path configs/l1_lpips_kl_gan.yaml \
  --vae f8d4 \
  --vae-ckpt pretrained/sdvae/sdvae-f8d4.pt \
  --disc-pretrained-ckpt pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt \
  --enc-type dinov2-vit-b \
  --proj-coeff 0.5 \
  --encoder-depth 8 \
  --batch-size 256 \
  --compile \
  --compile-mode max-autotune-no-cudagraphs
```
