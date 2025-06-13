---
title: "LEDITS++ 图像编辑：生成步骤"
date: 2025-06-13T16:35:01+08:00
draft: false
math: true
tags: ['Diffusion', 'LEDITS++', 'featured']
---

### 一、传统的扩散模型生成图像的步骤 

1.  **准备阶段 (Preparation)**
    *   **输入**: 一个文本提示词，例如 `"a photo of a cat"`。
    *   **操作**:
        *   将提示词编码成文本嵌入（text embeddings）。
        *   生成一个与目标图像尺寸相对应的、完全随机的噪声潜在变量$x_T$。

2.  **进入扩散循环 (Denoising Loop)**
    *   这是一个迭代`N`次（例如50次）的循环，从时间步$T$到$0$。

3.  **循环核心：预测并计算引导 (Predict & Guide)**
    *   在每一步$t$，UNet模型会进行**两次**前向传播，来预测噪声：
        *   一次是**有条件的 (conditional)**：使用你的文本嵌入（`"a photo of a cat"`）来预测噪声。
        *   一次是**无条件的 (unconditional)**：使用一个空的文本嵌入（`""`）来预测噪声。
    *   将两个预测结果根据引导系数（guidance scale）进行加权合并，得到最终的噪声预测`$\text{noise\_pred}$`。
        *   公式类似：`$ \text{noise\_pred} = \text{noise\_uncond} + \text{guidance\_scale} \times (\text{noise\_cond} - \text{noise\_uncond}) $`

4.  **循环核心：更新潜在变量 (Denoise Step)**
    *   使用上一步算出的最终噪声预测`$ \text{noise\_pred} $`和调度器（Scheduler），从当前的潜在变量$x_t$计算出下一步更清晰的潜在变量$x_{t-1}$。

5.  **结束循环并解码**
    *   循环结束后，得到最终的潜在变量$x_0$。
    *   使用VAE解码器将$x_0$转换成我们能看到的像素图像。



### 二、LEDITS++ 图像编辑步骤

**步骤0：[新] 反向过程 (Inversion - 对应创新点1)**
*   **输入**: 一张**真实**的图片（例如，Yann LeCun的照片）。
*   **操作**: 调用`pipe.invert()`。这个过程会模拟扩散的**反向**，找到一个特定的初始噪声$z$（论文中的$z_t$）和一系列中间潜在变量$x_t$，使得从这个噪声出发，可以完美地重建出原始输入图片。
*   **结果**: 我们不再从随机噪声开始，而是有了一个与原图绑定的“种子”噪声和路径。**这是整个编辑流程的前置准备，只做一次。**

**进入扩散循环 (Denoising Loop for Editing)**
*   这是一个与之前类似的迭代循环，但起点是`invert`算出的$x_T$。

**循环核心步骤：**

1.  **[改进] 并行预测多种噪声**
    *   在每一步$t$，UNet模型不再是做两次前向传播，而是**一次性做 `1 + N` 次**（其中N是编辑指令的数量）。
    *   对于样例输入，它会并行预测**三种**噪声：
        *   **无条件的** (基于 `""`)
        *   **基于 `'george clooney'` 的**
        *   **基于 `'sunglasses'` 的**
    *   这一步直接为**创新点2和3**提供了所有必需的计算原料。

2.  **[新] 为每个编辑概念计算“编辑向量”和“掩码”**
    *   这一步是针对每一个编辑指令独立进行的。

    *   **对于 `'george clooney'`**:
        *   **计算编辑向量 (创新点2)**: `$ \text{edit\_vector\_clooney} = \text{noise\_pred\_clooney} - \text{noise\_pred\_uncond} $`
        *   **计算掩码 (创新点3)**: `$ \text{final\_mask\_clooney} = M^1 \times M^2 $`
        *   **应用掩码**: `$ \text{masked\_vector\_clooney} = \text{edit\_vector\_clooney} \times \text{final\_mask\_clooney} $`
        
    *   **对于 `'sunglasses'`**:
        *   **计算编辑向量 (创新点2)**: `$ \text{edit\_vector\_sunglasses} = \text{noise\_pred\_sunglasses} - \text{noise\_pred\_uncond} $`
        *   **计算掩码 (创新点3)**: `$ \text{final\_mask\_sunglasses} = M^1 \times M^2 $`
        *   **应用掩码**: `$ \text{masked\_vector\_sunglasses} = \text{edit\_vector\_sunglasses} \times \text{final\_mask\_sunglasses} $`
    
3.  **[新] 合成最终引导 **
    *   将所有经过掩码和强度缩放的编辑向量加在一起。
    *   `$ \text{total\_guidance} = (\text{masked\_vector\_clooney} \times 3) + (\text{masked\_vector\_sunglasses} \times 4) $`
    *   最终的噪声预测为: `$ \text{final\_noise\_pred} = \text{noise\_pred\_uncond} + \text{total\_guidance} $`

4.  **更新潜在变量 (Denoise Step - 传统步骤)**
    *   使用上一步算出的`$ \text{final\_noise\_pred} $`和调度器（Scheduler），从$x_t$计算出$x_{t-1}$。

5.  **结束循环并解码 (传统步骤)**
    *   循环结束得到$x_0$，用VAE解码成最终编辑后的图像。

### 三、核心修改

把原来的 cfg 公式：

`$ \text{noise\_pred} = \text{noise\_uncond} + \text{guidance\_scale} \times (\text{noise\_cond} - \text{noise\_uncond}) $`

里面的

`$ \text{guidance\_scale} \times (\text{noise\_cond} - \text{noise\_uncond}) $`

改成为每个编辑词算一个编辑变量，并用掩码限制编辑区域：

`\begin{align*} 
\text{total\_guidance} &= \left( 3 \times \text{final\_mask\_clooney} \odot \left( \text{noise\_pred}\_{\text{clooney}} - \text{noise\_uncond} \right) \right) + \\ 
&\quad \left( 4 \times \text{final\_mask\_sunglasses} \odot \left( \text{noise\_pred}\_{\text{sunglasses}} - \text{noise\_uncond} \right) \right) 
\end{align*}`

### 四、主线代码流程

这里暂时不考虑掩码是怎么算的，就当已经得到了掩码了。后面会单独写一篇文章解释这个掩码是怎么设计的。

```python
def ledits_core_loop_simplified(
    unet,                           # UNet模型
    scheduler,                      # 调度器
    latents,                        # 初始潜在变量 x_T
    timesteps,                      # 时间步序列
    uncond_embeddings,              # 无条件文本嵌入 (对应 "")
    edit_concepts_embeddings,       # 编辑概念的文本嵌入列表 (如 ['george clooney', 'sunglasses'])
    edit_guidance_scales,           # 每个编辑概念的引导强度 (如 [3.0, 4.0])
    edit_masks,                     # 每个编辑概念的最终掩码 (假设已经计算好)
    reverse_editing_directions,     # 每个编辑概念的方向 (如 [False, False])
    edit_weights=None,              # 每个编辑概念的权重 (可选)
    verbose=True
):
    """
    LEDITS++ 核心去噪循环的简化版本
    
    主要创新点：
    1. 并行预测多种噪声 (1次UNet调用预测1+N种噪声)
    2. 为每个编辑概念独立计算编辑向量
    3. 使用掩码进行空间定位编辑
    4. 合成多个编辑概念的最终引导
    """
    
    # 获取编辑概念数量
    num_edit_concepts = len(edit_concepts_embeddings)
    print(f"开始LEDITS++去噪循环，共有 {num_edit_concepts} 个编辑概念")
    
    # 如果没有提供权重，默认所有概念权重相等
    if edit_weights is None:
        edit_weights = [1.0] * num_edit_concepts
    
    # ==================== 核心去噪循环 ====================
    for i, timestep in enumerate(timesteps):
        if verbose:
            print(f"\n--- 时间步 {i+1}/{len(timesteps)}, t={timestep} ---")
        
        # ==================== 步骤1: 并行预测多种噪声 ====================
        print("步骤1: 并行预测多种噪声")
        
        # 准备输入：将潜在变量复制 (1 + N) 次，用于并行预测
        # 第1个用于无条件预测，后N个用于各个编辑概念
        latent_model_input = torch.cat([latents] * (1 + num_edit_concepts))
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        
        # 准备文本嵌入：无条件 + 所有编辑概念
        text_embeddings = torch.cat([uncond_embeddings] + edit_concepts_embeddings)
        
        # 一次性UNet前向传播，并行预测所有噪声
        print(f"  - 输入形状: {latent_model_input.shape}")
        print(f"  - 文本嵌入形状: {text_embeddings.shape}")
        
        # 这里是关键：一次UNet调用预测1+N种噪声
        noise_pred_all = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
        
        # 分离不同的噪声预测
        noise_predictions = noise_pred_all.chunk(1 + num_edit_concepts)
        noise_pred_uncond = noise_predictions[0]  # 无条件噪声预测
        noise_pred_edit_concepts = noise_predictions[1:]  # 各编辑概念的噪声预测
        
        print(f"  - 无条件噪声预测形状: {noise_pred_uncond.shape}")
        print(f"  - 编辑概念噪声预测数量: {len(noise_pred_edit_concepts)}")
        
        # ==================== 步骤2: 为每个编辑概念计算编辑向量和应用掩码 ====================
        print("步骤2: 为每个编辑概念计算编辑向量和应用掩码")
        
        # 存储每个概念的最终编辑向量
        masked_edit_vectors = []
        
        for c, noise_pred_edit in enumerate(noise_pred_edit_concepts):
            concept_name = f"概念{c+1}"  # 实际应用中可以传入概念名称
            print(f"  处理 {concept_name}:")
            
            # 2.1 计算原始编辑向量 (创新点2)
            edit_vector = noise_pred_edit - noise_pred_uncond
            print(f"    - 计算编辑向量: noise_pred_edit - noise_pred_uncond")
            
            # 2.2 应用编辑方向
            if reverse_editing_directions[c]:
                edit_vector = edit_vector * -1
                print(f"    - 应用反向编辑方向")
            
            # 2.3 应用引导强度缩放
            edit_vector = edit_vector * edit_guidance_scales[c]
            print(f"    - 应用引导强度: {edit_guidance_scales[c]}")
            
            # 2.4 应用掩码进行空间定位 (创新点3)
            # 这里假设edit_masks[c]已经是计算好的最终掩码
            masked_edit_vector = edit_vector * edit_masks[c]
            print(f"    - 应用空间掩码，掩码形状: {edit_masks[c].shape}")
            
            # 2.5 应用概念权重
            masked_edit_vector = masked_edit_vector * edit_weights[c]
            print(f"    - 应用概念权重: {edit_weights[c]}")
            
            masked_edit_vectors.append(masked_edit_vector)
        
        # ==================== 步骤3: 合成最终引导 ====================
        print("步骤3: 合成最终引导")
        
        # 初始化总引导为零
        total_guidance = torch.zeros_like(noise_pred_uncond)
        
        # 累加所有编辑概念的贡献
        for c, masked_vector in enumerate(masked_edit_vectors):
            total_guidance = total_guidance + masked_vector
            print(f"  - 累加概念{c+1}的编辑向量")
        
        # 计算最终的噪声预测
        final_noise_pred = noise_pred_uncond + total_guidance
        print(f"  - 最终噪声预测 = 无条件预测 + 总引导")
        
        # ==================== 步骤4: 更新潜在变量 (传统去噪步骤) ====================
        print("步骤4: 更新潜在变量")
        
        # 使用调度器进行去噪步骤: x_t -> x_{t-1}
        latents = scheduler.step(final_noise_pred, timestep, latents).prev_sample
        print(f"  - 潜在变量更新完成，新形状: {latents.shape}")
    
    print("\n==================== 去噪循环完成 ====================")
    return latents
```

