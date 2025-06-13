---
title: "从输入到输出: UNet in Stable Diffusion"
date: 2025-06-13T20:14:03+08:00
draft: false
tags: ['Diffusion','UNet', 'featured']
math: true
---

UNet in Stable Diffusion 接受三个输入，分别是：
- 带噪点的潜空间图像，维度是 `(1, 4, 64, 64)`
- 时间步，维度是 `(1,)`
- 文本提示编码，维度是 `(1, 77, 768)`

这三个输入经过 UNet 后，会得到一个输出，维度是 `(1, 4, 64, 64)`。

UNet 的结构可以分为五个部分：
- Initial Conv
- Encoder
- Middle Block
- Decoder
- Final Conv

下面是输入经过这些层后，维度是如何变化的：

### 结构分析


#### **第一部分：Initial Conv**

这其实就是一个普通的卷积层，把输入的维度从 `(1, 4, 64, 64)` 变成了 `(1, 320, 64, 64)`。写成代码就是一行：

```python
Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

#### **第五部分：Final Conv**

这也是一个普通的卷积层，把输入的维度从 `(1, 320, 64, 64)` 变成了 `(1, 4, 64, 64)`。写成代码也是一行：

```python
Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```

所以剩下的三个部分，其实都是在维度 `(1, 320, 64, 64)` 的基础上操作的。

#### **第二部分：Encoder**

Encoder 包含两类Block:
- CrossAttnDownBlock2D
- DownBlock2D

他俩的核心区别在于：**`CrossAttnDownBlock2D` 额外包含了一个 `Transformer2DModel`，这个组件专门用来执行自注意力和交叉注意力计算。**


| 对比指标 | Down Block 1 | Down Block 2 | Down Block 3 | Down Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `DownBlock2D` |
| **尺寸变化** | 64x64 -> 32x32 | 32x32 -> 16x16 | 16x16 -> 8x8 | 8x8|
| **通道数变化** | 320 | 320 -> 640 | 640 -> 1280 | 1280  |
| **ResNet 数量** | 2 | 2 | 2 | 2 |
| **Transformer 数量**| 2 | 2 | 2 | **0** |
| **内部数据流** | `(ResNet -> Transformer)` x 2 交错循环 | `(ResNet -> Transformer)` x 2 交错循环 | `(ResNet -> Transformer)` x 2 交错循环 | `ResNet` -> `ResNet` 线性序列 |
| **主要功能** | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li><li>**空间降采样**</li> | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li><li>**空间降采样**</li> | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li><li>**空间降采样**</li> | <li>特征提取</li> |
| **主要输入** | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`encoder_hidden_states` (文本)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`encoder_hidden_states` (文本)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`encoder_hidden_states` (文本)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li> |



#### **第三部分：Middle Block**

*   **输入 `x`**: `(1, 1280, 8, 8)`
*   **内部层**: 1 x **ResBlock**, 1 x **Attention Block**, 1 x **ResBlock**。
*   **输出 `x`**: `(1, 1280, 8, 8)`

---

#### **第四部分：Decoder**

| 对比指标 | Up Block 1 | Up Block 2 | Up Block 3 | Up Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `UpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` |
| **尺寸变化** | 8x8 -> 16x16 | 16x16 -> 32x32 | 32x32 -> 64x64 | 64x64 -> 64x64 |
| **通道数变化** | 1280 + 1280 -> 1280 | 1280 + 1280 -> 640 | 640 + 640 -> 320 | 320 + 320 -> 320 |
| **ResNet 数量** | 3 | 3 | 3 | 3 |
| **Transformer 数量**| **0** | 3 | 3 | 3 |
| **内部数据流** | `(拼接 -> ResNet)` x 3 | `(拼接 -> ResNet -> Transformer)` x 3 | `(拼接 -> ResNet -> Transformer)` x 3 | `(拼接 -> ResNet -> Transformer)` x 3 |
| **主要功能** | <li>特征提取</li><li>**空间上采样**</li> | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li><li>**空间上采样**</li> | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li><li>**空间上采样**</li> | <li>特征提取</li><li>全局上下文理解</li><li>文本指令融合</li> |
| **主要输入** | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`res_hidden_states` (跨层连接)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`res_hidden_states` (跨层连接)</li><li>`encoder_hidden_states` (文本)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`res_hidden_states` (跨层连接)</li><li>`encoder_hidden_states` (文本)</li> | <li>`hidden_states` (图像)</li><li>`temb` (时间)</li><li>`res_hidden_states` (跨层连接)</li><li>`encoder_hidden_states` (文本)</li> |


#### **3. 维度之旅：在 `forward` 方法的执行体中**

`forward` 方法内部的执行流程，完美复现了我描述的维度之旅。

-   **时间`t`的处理**:
    ```python
    # 1. time
    t_emb = self.time_proj(timesteps)
    emb = self.time_embedding(t_emb, timestep_cond)
    ```
    这里将数字`t`转换为了时间嵌入向量`emb`，维度是 `(1, 320)`，这个`emb`会被传递给后续所有的ResNet模块。

-   **左侧下行（编码器）**:
    ```python
    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        ...
        sample, res_samples = downsample_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            ...
        )
        down_block_res_samples += res_samples
    ```
    -   `for downsample_block in self.down_blocks:` 这个循环就是我说的“下行之旅”。
    -   `temb=emb` 就是把时间`t`的嵌入加入。
    -   `encoder_hidden_states=encoder_hidden_states` 就是把文本`context`注入（用于交叉注意力）。
    -   `res_samples` 就是每一层输出的、用于“天桥”（Skip Connection）的特征图。`down_block_res_samples += res_samples` 这行代码就是把它们全部收集起来。

-   **底部（瓶颈）**:
    ```python
    # 4. mid
    if self.mid_block is not None:
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            ...
        )
    ```
    这就是在最底层进行处理。

-   **右侧上行（解码器）**:
    ```python
    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        sample = upsample_block(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=res_samples,
            encoder_hidden_states=encoder_hidden_states,
            ...
        )
    ```
    -   `for ... in self.up_blocks:` 这个循环就是“上行之旅”。
    -   `res_samples = down_block_res_samples[...]` 这两行代码，就是在从收集好的“天桥”数据中，从后往前取出对应层的特征图。
    -   `res_hidden_states_tuple=res_samples` 就是把取出的特征图（通过天桥过来的）传递给上行模块，用于拼接。
