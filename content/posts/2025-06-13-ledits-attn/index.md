---
title: "LEDITS++: Attention Map"
date: 2025-06-13T19:57:38+08:00
draft: false
tags: ['Diffusion', 'LEDITS++', 'featured']
math: true
---

## 1. UNet中的注意力机制基础

在Stable Diffusion的UNet中，有两种主要的注意力机制：

### 自注意力（Self-Attention）
- **作用**：让图像的不同区域之间相互关注
- **输入**：图像特征本身
- **目的**：捕捉图像内部的空间关系

### 交叉注意力（Cross-Attention）
- **作用**：让图像特征与文本特征进行交互
- **输入**：Query来自图像特征，Key和Value来自文本embedding
- **目的**：根据文本描述来指导图像生成

## 2. LEDITS的注意力修改

### 关键组件分析：

#### `CrossAttnProcessor` 类
```python
class CrossAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, ...):
        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # 关键：存储注意力分数用于后续处理
        self.attnstore(attention_probs,
                       is_cross=True,
                       place_in_unet=self.place_in_unet,
                       editing_prompts=self.editing_prompts,
                       PnP=self.PnP)
```

#### `prepare_unet` 方法
```python
def prepare_unet(self, attention_store, PnP: bool = False):
    for name in self.unet.attn_processors.keys():
        if "attn2" in name and place_in_unet != 'mid':  # attn2 = 交叉注意力
            # 只对交叉注意力层使用自定义处理器
            attn_procs[name] = CrossAttnProcessor(...)
        else:
            # 自注意力层使用默认处理器
            attn_procs[name] = AttnProcessor()
```

## 3. 注意力图的处理

### Token级别的注意力
每个prompt中的每个token都会产生独立的注意力图：

```python
# 从存储的注意力中提取特定editing prompt的注意力图
out = self.attention_store.aggregate_attention(
    attention_maps=self.attention_store.step_store,
    prompts=self.text_cross_attention_maps,
    res=16,  # 16x16分辨率
    from_where=["up", "down"],  # UNet的上采样和下采样部分
    is_cross=True,
    select=self.text_cross_attention_maps.index(editing_prompt[c]),
)

# 提取除了startoftext token之外的所有token的注意力
attn_map = out[:, :, :, 1:1 + num_edit_tokens[c]]  # 0 -> startoftext
```

### 多Token的合并策略
对于包含多个token的prompt，LEDITS采用**求和平均**的策略：

```python
# 对所有token的注意力图求和（平均）
assert (attn_map.shape[3] == num_edit_tokens[c])
attn_map = torch.sum(attn_map, dim=3)  # 在token维度上求和
```

## 4. 注意力图的后处理

### 高斯平滑
```python
# 对注意力图进行高斯平滑，减少噪声
attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
attn_map = self.smoothing(attn_map).squeeze(1)
```

### 二值化掩码生成
```python
# 根据阈值创建二值掩码
tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold_c, dim=1)
attn_mask = torch.where(attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1,16,16), 1.0, 0.0)

# 将掩码从16x16上采样到64x64（潜在空间分辨率）
attn_mask = F.interpolate(
    attn_mask.unsqueeze(1),
    noise_guidance_edit_tmp.shape[-2:]  # 64,64
).repeat(1, 4, 1, 1)  # 重复4次对应4个潜在通道
```

## 5. 完整流程总结

1. **注意力捕获**：`CrossAttnProcessor`在每个去噪步骤中捕获交叉注意力分数
2. **注意力聚合**：`aggregate_attention`提取特定editing prompt的注意力图
3. **Token合并**：对多个token的注意力图求和平均
4. **后处理**：高斯平滑 + 阈值化生成二值掩码
5. **应用掩码**：将掩码应用到噪声引导上，实现局部编辑

## 关键理解点：

- **每个token都有独立的注意力图**，但最终会合并
- **注意力图分辨率是16x16**，然后上采样到64x64匹配潜在空间
- **只修改交叉注意力**，自注意力保持不变
- **掩码用于控制编辑的空间范围**，只在注意力高的区域应用编辑

这样的设计让LEDITS能够精确控制编辑发生的位置，避免影响图像的其他部分。