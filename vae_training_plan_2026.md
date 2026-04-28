# 面向 ImageNet SOTA 的 SiT+VAE 端到端联合微调计划 (2026 REPA-E 架构版)

基于最新的开源库大搜查，我们发现了与你当前架构 100% 契合的宝藏级框架：**`End2End-Diffusion/REPA-E`**。
该框架原生地支持 **SiT + VAE 的端到端 (E2E) 联合微调**，并且自带了 L1 + LPIPS + PatchGAN 的极其规范的分布式训练代码。

本计划的核心目标是：**直接采用 REPA-E 作为工程基座，在其中注入我们创新的“带全局自适应自由度的 Student-t 多变量先验”，一站式解决 VAE 和 SiT 的联合 SOTA 训练问题。**

---

## 1. 核心损失函数配方 (The SOTA Recipe)

由于我们直接使用了 REPA-E 的损失模块骨架，我们将原生地获得以下最佳实践：

### A. 基础重建损失 (Reconstruction Loss)
- **L1 Loss**：取代导致模糊的 L2，更好地保留锐利边缘。

### B. 感知与对抗损失 (Perceptual & Adversarial Loss)
- **LPIPS (感知损失)**：利用 `timm` 的预训练网络比较深层特征差异。
- **PatchGAN Discriminator (对抗损失)**：带有自适应权重（Adaptive Weighting）的局部判别器，强制 Decoder 生成 1.0 级别 FID 的逼真纹理。

### C. 先验对齐损失 (Latent Prior Loss) - 【核心修改点】
- **全局自适应自由度 (Learnable $\nu$)**：我们将原本固定的自由度设为可学习的全局参数（`nn.Parameter`），让模型根据 ImageNet 数据密度自动寻找最佳的长尾结构。
- **多变量 Student-t 避免“长满刺的球”**：我们构建的是基于整个范数平方 $\|z\|^2$ 的**多变量 Student-t (Multivariate Student-t)**。它在高维空间中严格保持**球面对称**（具有高斯的完全梯度包容性），绝不是各维度独立的、长满刺的星芒状结构。
- **带 $\nu$ 梯度的 KL 散度 (MC 估计)**：由于 $\nu$ 可学习，必须加入配分常数项 `lgamma`。
  
  **具体构建方法（将注入到 REPA-E 的 Loss 中）**：
  ```python
  # 1. 全局自适应自由度参数定义 (将初始化为 29.0)
  # self.nu_raw = nn.Parameter(torch.tensor(29.0))
  nu = torch.nn.functional.softplus(self.nu_raw) + 1e-4 
  d = z.shape[1] * z.shape[2] * z.shape[3] # 潜空间总维度 (如 4096)
  
  # 2. 负高斯熵
  entropy_term = -0.5 * log_var.float().flatten(1).sum(dim=1)
  
  # 3. 多变量 Student-t 惩罚 (基于 ||z||^2 保证球面对称)
  z_norm_sq = z.float().square().flatten(1).sum(dim=1)
  prior_penalty = ((nu + d) / 2.0) * torch.log(1.0 + z_norm_sq / nu)
  
  # 4. 配分常数项 (为了计算 nu 的梯度)
  log_gamma_term = - torch.lgamma((nu + d) / 2.0) + torch.lgamma(nu / 2.0) + (d / 2.0) * torch.log(nu)
  
  # 结合 REPA-E 的低权重缩放 (beta)
  kl_loss = (entropy_term + prior_penalty + log_gamma_term).mean()
  ```

---

## 2. 工程实现框架：REPA-E 端到端架构

我们不再从零手写训练脚本，而是**全面克隆并接管 `REPA-E` 的 `train_repae.py`**，其先天优势如下：

1. **三阶并发优化器**：
   框架内部已经完全解耦了三个优化器：`SiT 优化器`、`VAE 优化器` 和 `Discriminator 优化器`。这允许我们在同一次前向传播中，同时拉齐 VAE 的分布和 SiT 的生成质量。
2. **SiT 基座完全对齐**：
   他们验证所用的扩散模型本身就是 `SiT-XL/2`，这与你的项目毫无代差，Latent 的 Normalization 逻辑开箱即用。
3. **🔥 关键训练精度控制 (Mixed Precision 避坑指南)**：
   - REPA-E 原生通过 `Hugging Face Accelerate` 提供全局 `BF16 (Bfloat16)` 加速。
   - 我们只需要在写入 Student-t 损失的地方（如上方代码所示），严格保留对 `log_var.float()` 和 `z.float()` 的 Upcast 强制升维操作，防止指数计算导致的 `NaN` 爆炸。

---

## 3. 执行路线图与下一步任务

该路线图已经达到了可以直接动手编码执行的程度：

### 阶段一：获取并剥离核心代码 (Git Clone & Copy)
- [ ] 从 `https://github.com/End2End-Diffusion/REPA-E.git` 克隆核心训练代码。
- [ ] 将其 `train_repae.py` 以及核心的 `loss/` 和 `models/` 目录移植到你当前的 `SiT/` 工作空间中。

### 阶段二：外科手术式代码魔改 (Injection)
- [ ] **Loss 魔改**：打开 `loss/losses.py`，定位到传统 Gaussian KL 散度的计算位置，将其完全替换为我们的 **Student-t 散度代码块**。
- [ ] **网络魔改**：在负责计算 Loss 的 Module 中，注册 `self.nu_raw = nn.Parameter(torch.tensor(29.0))`，让其被 `optimizer_vae` 捕获并参与反向传播。

### 阶段三：端到端小规模试跑 (Validation)
- [ ] 在 ImageNet 的一个小规模子集上启动修改后的 `train_repae.py`。
- [ ] **绝密监控指标**：
  1. 监控 `kl_loss` 是否平稳下降（证明 FP32 升维和 `lgamma` 梯度工作正常）。
  2. 打印并监控全局参数 $\nu$ (`self.nu_raw`) 的变化趋势。看它是保持在 29 附近，还是根据 ImageNet 的复杂度被自适应地推向了更小的重尾值！

---
*此计划已全面修订，基座切换至 REPA-E。随时可以拉取代码开始第一刀的外科手术式魔改！*
