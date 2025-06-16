---
title: "Stable Diffusion Attention 分析"
date: 2025-06-13T20:06:31+08:00
draft: false
tags: ['Diffusion', 'Attention', 'featured']
math: true
---

许多基于文生图模型的图像编辑方法，例如 prompt-to-prompt，ledits++ 等，都会利用 Stable Diffusion 的 UNet 里的注意力机制，来帮助定位编辑词在图像中的位置，从而实现文本引导的图像编辑。

本文会说明 Stable Diffusion UNet 里的注意力具体是怎么算的。

UNet 模型里，注意力机制是通过 Transformer 来实现的，主要在以下几个地方：
- Encoder 里有3个 CrossAttnDownBlock2D，每个 CrossAttnDownBlock2D 包含 2个 Transformer。
- Middle 里包含 1个 Transformer。
- Decoder 里有3个 CrossAttnUpBlock2D，每个 CrossAttnUpBlock2D 包含 3个 Transformer。


本文会以 Encoder 里的第一个 Transformer2DModel 为例，说明注意力机制的具体计算过程。

### Transformer2DModel 计算流程

此表描述了**从一个 4D 图像特征图输入，到经过 `Transformer2DModel` 处理后，输出一个新的 4D 图像特征图的完整过程。**

**前提**:
-   **输入 `hidden_states`**: `[B, 320, 64, 64]`
-   **输入 `encoder_hidden_states`**: `[B, 77, 768]`

| 步骤 | 操作 / 模块 | 输入(s) 及其维度 | 关键层定义 (in -> out) | 输出维度 | 解释 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.1**| **保存残差** | `hidden_states`: `[B, 320, 64, 64]` | `residual = hidden_states` | `[B, 320, 64, 64]` | 保存最原始的输入，用于最后的残差连接。 |
| **1.1**| `self.norm` | `hidden_states`: `[B, 320, 64, 64]` | `GroupNorm(32, 320)` | `[B, 320, 64, 64]` | 对图像特征进行归一化。 |
| **1.2**| `self.proj_in` | `(上一步输出)`: `[B, 320, 64, 64]` | `Conv2d(320 -> 320)` | `[B, 320, 64, 64]` | 输入投影 (1x1卷积)。 |
| **1.3**| **Reshape** | `(上一步输出)`: `[B, 320, 64, 64]` | `permute & reshape` | `[B, 4096, 320]` | **从图像域转为序列域**，为送入注意力模块做准备。|
| **2.0**| **`self.transformer_blocks` (循环)** | `(上一步输出)`: `[B, 4096, 320]` <br> `encoder_hidden_states`: `[B, 77, 768]` | `for block in ...` | `[B, 4096, 320]` | **调用 `BasicTransformerBlock`**，执行自注意力、交叉注意力和前馈网络等核心计算。 |
| **3.1**| **Reshape** | `(上一步输出)`: `[B, 4096, 320]` | `reshape & permute` | `[B, 320, 64, 64]` | **从序列域恢复为图像域**。 |
| **3.2**| `self.proj_out` | `(上一步输出)`: `[B, 320, 64, 64]` | `Conv2d(320 -> 320)` | `[B, 320, 64, 64]` | 输出投影 (1x1卷积)。 |
| **3.3**| **残差连接** | `(上一步输出)` + `residual` | `+` | **`[B, 320, 64, 64]`** | **核心！** 将处理结果与原始输入相加，得到最终输出。|

### BasicTransformerBlock 计算流程

此表描述了**从一个 3D 序列输入，到经过 `BasicTransformerBlock` 处理后，输出一个新的 3D 序列的完整过程。**

**前提**:
-   **输入 `hidden_states`**: `[B, 4096, 320]` (来自 `Transformer2DModel` 的预处理)
-   **输入 `encoder_hidden_states`**: `[B, 77, 768]`

| 阶段 | 步骤 | 操作 / 模块 | 输入(s) 及其维度 | 输出维度 | 源代码对应 (简化) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **自注意力 (Self-Attention)** | 1.1 | **归一化 (Pre-Norm)** | `hidden_states`: `[B, 4096, 320]` | `[B, 4096, 320]` | `norm_hidden_states = self.norm1(hidden_states)` |
| | 1.2 | **注意力计算** | `(上一步输出)`: `[B, 4096, 320]` | `[B, 4096, 320]` | `attn_output = self.attn1(norm_hidden_states, ...)` |
| | 1.3 | **残差连接** | `(步骤1.2输出)`: `[B, 4096, 320]` <br> `hidden_states`: `[B, 4096, 320]` | `[B, 4096, 320]` | `hidden_states = attn_output + hidden_states` |
| **交叉注意力 (Cross-Attention)** | 2.1 | **归一化 (Pre-Norm)** | `(上一步输出)`: `[B, 4096, 320]` | `[B, 4096, 320]` | `norm_hidden_states = self.norm2(hidden_states)` |
| | 2.2 | **注意力计算** | **Q**: `(上一步输出)` `[B, 4096, 320]`<br>**K,V**: `encoder_hidden_states` `[B, 77, 768]`| `[B, 4096, 320]` | `attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, ...)` |
| | 2.3 | **残差连接** | `(步骤2.2输出)`: `[B, 4096, 320]` <br> `(步骤1.3输出)`: `[B, 4096, 320]`| `[B, 4096, 320]` | `hidden_states = attn_output + hidden_states` |
| **前馈网络 (Feed-Forward)** | 3.1 | **归一化 (Pre-Norm)** | `(上一步输出)`: `[B, 4096, 320]` | `[B, 4096, 320]` | `norm_hidden_states = self.norm3(hidden_states)` |
| | 3.2 | **前馈网络计算** | `(上一步输出)`: `[B, 4096, 320]` | `[B, 4096, 320]` | `ff_output = self.ff(norm_hidden_states)` |
| | 3.3 | **残差连接** | `(步骤3.2输出)`: `[B, 4096, 320]` <br> `(步骤2.3输出)`: `[B, 4096, 320]`| **`[B, 4096, 320]`** | `hidden_states = ff_output + hidden_states` |

**补充**：

1. 在每个子模块（`attn1`, `attn2`, `ff`）进行实际计算**之前**，都会先对其输入进行一次 `LayerNorm` (`norm1`, `norm2`, `norm3`)。这有助于稳定训练过程。
2. 残差连接的输入源:
    *   **步骤 1.3**: `attn1` 的输出是与**最原始的输入** `hidden_states` 相加。
    *   **步骤 2.3**: `attn2` 的输出是与**第1阶段（自注意力阶段）完成后的结果**相加。
    *   **步骤 3.3**: `ff` 的输出是与**第2阶段（交叉注意力阶段）完成后的结果**相加。
    *   每一阶段的输出都作为下一阶段的输入，这种链式结构+残差连接保证了信息的逐层处理和传递。
3. 输入与输出: `BasicTransformerBlock` 接收一个 `[B, SeqLen, Dim]` 的张量作为主要输入，经过内部三个阶段的处理后，输出一个**形状完全相同**的 `[B, SeqLen, Dim]` 张量。它是一个“就地升级”模块，只改变特征的“内容”，不改变其“形状”。

### attn1 计算流程

**此表描述了 `attn1` 如何处理图像特征，让图像的各个部分相互“沟通”，理解内部结构。**

**前提**:
-   **输入 `hidden_states`**: `[B, 4096, 320]` (这是经过 `norm1` 归一化后的图像特征)
-   **配置**: `num_attention_heads=8`, `attention_head_dim=40` (因此 `inner_dim=320`)

| 步骤 | 操作 | 输入(s) 及其维度 | 关键层定义 (in -> out) | 输出维度 | 解释 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **生成 Q, K, V** | `hidden_states`: `[B, 4096, 320]` | `to_q`: `Linear(320 -> 320)` <br> `to_k`: `Linear(320 -> 320)` <br> `to_v`: `Linear(320 -> 320)` | **Q, K, V**: `[B, 4096, 320]` | **Q, K, V 全部来自同一个输入源**，即图像特征本身。 |
| 2 | **多头拆分** | `Q, K, V`: `[B, 4096, 320]` | `reshape & permute` | **Q, K, V**: `[B*8, 4096, 40]` | 将 320 维特征拆分为 8 个 40 维的“头”，并行处理。 |
| 3 | **计算注意力**| `Q, K, V` | **`self.processor(...)`** | `[B*8, 4096, 40]` | **调用 AttnProcessor**。它内部完成 QK<sup>T</sup> 矩阵相乘、缩放、Softmax 和与 V 的加权聚合。 |
| 4 | **多头合并** | `(上一步输出)`: `[B*8, 4096, 40]` | `reshape & permute` | `[B, 4096, 320]` | 将 8 个“头”的结果拼接起来，恢复为 320 维。 |
| 5 | **输出投影** | `(上一步输出)`: `[B, 4096, 320]` | `to_out`: `Linear(320 -> 320)` | **`[B, 4096, 320]`** | 对合并后的特征进行一次线性变换，得到最终输出。 |

### attn2 计算流程

**此表描述了 `self.attn2` 如何将文本信息注入到图像特征中，实现条件控制。**

**前提**:
-   **输入 `hidden_states` (for Q)**: `[B, 4096, 320]` (这是经过 `norm2` 归一化后的图像特征)
-   **输入 `encoder_hidden_states` (for K,V)**: `[B, 77, 768]` (来自文本编码器)
-   **配置**: `num_attention_heads=8`, `attention_head_dim=40`, `cross_attention_dim=768`

| 步骤 | 操作 | 输入(s) 及其维度 | 关键层定义 (in -> out) | 输出维度 | 解释 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **生成 Q, K, V** | **Q**: `hidden_states`: `[B, 4096, 320]` <br> **K,V**: `encoder_hidden_states`: `[B, 77, 768]` | `to_q`: `Linear(320 -> 320)` <br> `to_k`: `Linear(768 -> 320)` <br> `to_v`: `Linear(768 -> 320)` | **Q**: `[B, 4096, 320]` <br> **K,V**: `[B, 77, 320]` | **Q 来自图像，K 和 V 来自文本**，这是交叉注意力的核心。 |
| 2 | **多头拆分** | `Q`: `[B, 4096, 320]` <br> `K, V`: `[B, 77, 320]`| `reshape & permute` | **Q**: `[B*8, 4096, 40]` <br> **K,V**: `[B*8, 77, 40]` | 将图像和文本的特征分别拆分为 8 个头。 |
| 3 | **计算注意力**| `Q, K, V` | **`self.processor(...)`** | `[B*8, 4096, 40]` | **调用 AttnProcessor**。它内部完成 QK<sup>T</sup> 矩阵相乘（得到 `4096x77` 的注意力图）、缩放、Softmax 和与 V 的加权聚合。 |
| 4 | **多头合并** | `(上一步输出)`: `[B*8, 4096, 40]` | `reshape & permute` | `[B, 4096, 320]` | 将 8 个“头”的结果拼接起来。 |
| 5 | **输出投影** | `(上一步输出)`: `[B, 4096, 320]` | `to_out`: `Linear(320 -> 320)` | **`[B, 4096, 320]`** | 对融合了文本信息的新图像特征进行线性变换。 |



### **`AttnProcessor` 的作用和位置**


**`AttnProcessor` 不在 `Transformer2DModel` 层面，也不在 `BasicTransformerBlock` 层面，而是在最底层的 `Attention` 模块（即 `attn1` 和 `attn2`）内部被调用。**

`Attention` 模块本身不决定如何计算注意力，而是把这个任务**委托**给它内部的一个 `processor` 对象。`AttnProcessor`就像一个**可替换的“计算引擎”**。


整个调用关系：

`Transformer2DModel.forward()`
&nbsp;&nbsp;&nbsp;&nbsp;`->` `BasicTransformerBlock.forward()`
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`->` `Attention.forward()` (例如调用 `self.attn1`)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`->` **`self.processor(...)`**  **<-- AttnProcessor 在这里被执行！**


主要有两种 `processor`：

1.  **`AttnProcessor` (默认)**:
    *   **实现方式**: 使用纯 PyTorch 的标准操作（`torch.bmm`, `softmax` 等）来实现注意力计算。
    *   **特点**: 兼容性好，但在速度和显存占用上不是最优的。
    *   **流程**: 它会执行我们之前在 `attn1`/`attn2` 表格中分解的所有步骤（生成QKV、矩阵相乘、Softmax、加权聚合等）。

2.  **`XFormersAttnProcessor` (优化)**:
    *   **来源**: 需要安装 `xformers` 库后手动设置 (`pipe.enable_xformers_memory_efficient_attention()`)。
    *   **实现方式**: 它会调用 `xformers` 库中高度优化的 `memory_efficient_attention` 函数。
    *   **特点**: **速度更快，显存占用显著降低**。它将多个步骤（如矩阵相乘、Softmax、加权聚合）融合成一个单一的、高效的 CUDA kernel 来执行，避免了生成巨大的注意力矩阵 (`[B*8, 4096, 4096]`)，从而节省了大量显存。




**attn 2 的内部详细流程**

| 步骤 | 操作 / 模块 | 输入(s) 及其维度 | 关键层/方法定义 | 输出维度 | 源代码对应 (简化) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **保存残差** | `hidden_states`: `[B, 4096, 320]` | - | `[B, 4096, 320]` | `residual = hidden_states` |
| **1** | **生成 Query** | `hidden_states`: `[B, 4096, 320]` | `attn.to_q`: `Linear(320 -> 320)` | `[B, 4096, 320]` | `query = attn.to_q(hidden_states)` |
| **2** | **准备 K,V 的源** | `encoder_hidden_states`: `[B, 77, 768]`| (可选 `norm_cross`) | `[B, 77, 768]` | `encoder_hidden_states = ...` |
| **3** | **生成 Key** | `encoder_hidden_states`: `[B, 77, 768]` | `attn.to_k`: `Linear(768 -> 320)` | `[B, 77, 320]` | `key = attn.to_k(encoder_hidden_states)` |
| **4** | **生成 Value** | `encoder_hidden_states`: `[B, 77, 768]` | `attn.to_v`: `Linear(768 -> 320)` | `[B, 77, 320]` | `value = attn.to_v(encoder_hidden_states)` |
| **5** | **多头拆分 Q** | `query`: `[B, 4096, 320]` | `attn.head_to_batch_dim` | `[B*8, 4096, 40]` | `query = ...` |
| **6** | **多头拆分 K** | `key`: `[B, 77, 320]` | `attn.head_to_batch_dim` | `[B*8, 77, 40]` | `key = ...` |
| **7** | **多头拆分 V** | `value`: `[B, 77, 320]` | `attn.head_to_batch_dim` | `[B*8, 77, 40]` | `value = ...` |
| **8**| **计算注意力分数** | **Q**: `[B*8, 4096, 40]` <br> **K**: `[B*8, 77, 40]` | `attn.get_attention_scores` | `[B*8, 4096, 77]` | 内部完成 QK<sup>T</sup>, 缩放, Softmax |
| **9**| **加权聚合** | **Probs**: `[B*8, 4096, 77]` <br> **V**: `[B*8, 77, 40]` | `torch.bmm` | `[B*8, 4096, 40]` | `hidden_states = torch.bmm(attention_probs, value)` |
| **10**| **多头合并** | `(上一步输出)`: `[B*8, 4096, 40]` | `attn.batch_to_head_dim` | `[B, 4096, 320]` | `hidden_states = ...` |
| **11**| **输出投影** | `(上一步输出)`: `[B, 4096, 320]` | `attn.to_out[0]`: `Linear(320 -> 320)` | `[B, 4096, 320]` | `hidden_states = attn.to_out[0](hidden_states)` |
| **12**| **Dropout** | `(上一步输出)`: `[B, 4096, 320]` | `attn.to_out[1]` | `[B, 4096, 320]` | `hidden_states = attn.to_out[1](hidden_states)` |
| **13**| **残差连接** | `(上一步输出)`: `[B, 4096, 320]` <br> `residual`: `[B, 4096, 320]` | `+` | **`[B, 4096, 320]`** | `hidden_states = hidden_states + residual` |



### 替换 AttnProcessor

得到 Stable Diffusion UNet 以后，有一个常见的方法，就是替换其中的 AttnProcessor，从而可以实现保存注意力图，或者修改注意力图，从而实现图像编辑。

下面解释清楚这个方法具体怎么做。

下面是一个默认的 AttnProcessor 的代码：
```python
class AttnProcessor:  # 定义一个名为 AttnProcessor 的类
    r"""
    用于执行注意力相关计算的默认处理器。
    """

    def __call__(  # 定义该类的可调用方法，使其像函数一样被调用
        self,
        attn: Attention,  # 传入的 Attention 模块实例
        hidden_states,  # 输入的隐藏状态张量 (通常是图像特征)
        encoder_hidden_states=None,  # 用于交叉注意力的编码器隐藏状态，如果为 None 则是自注意力
        attention_mask=None,  # 注意力掩码，用于忽略某些位置的计算
        temb=None,  # 时间步嵌入 (Time Embedding)，通常用于条件扩散模型
    ):
        residual = hidden_states  # 保存原始的 hidden_states，用于之后的残差连接

        if attn.spatial_norm is not None:  # 检查是否存在空间归一化层 (spatial_norm)
            hidden_states = attn.spatial_norm(
                hidden_states, temb
            )  # 如果存在，则应用空间归一化

        input_ndim = hidden_states.ndim  # 获取输入 hidden_states 张量的维度数

        if (
            input_ndim == 4
        ):  # 如果输入是4维张量 (通常是 [batch, channel, height, width])
            batch_size, channel, height, width = (
                hidden_states.shape
            )  # 获取张量的各个维度大小
            hidden_states = hidden_states.view(  # 将4D张量重塑为3D
                batch_size, channel, height * width
            ).transpose(
                1, 2
            )  # 并交换第1和第2个维度，变为 [batch, sequence_length, channel]

        # 根据是自注意力还是交叉注意力，获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape  # 如果是自注意力，从 hidden_states 获取形状
            if encoder_hidden_states is None  # 判断条件：encoder_hidden_states 是否为空
            else encoder_hidden_states.shape  # 如果是交叉注意力，从 encoder_hidden_states 获取形状
        )
        # 准备注意力掩码，确保其形状和设备与计算兼容
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:  # 检查是否存在组归一化层 (group_norm)
            # 应用组归一化。需要先将通道维度换回来，归一化后再换回去
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(
            hidden_states
        )  # 通过一个线性层将 hidden_states 映射为 Query (Q)

        if encoder_hidden_states is None:  # 如果是自注意力 (self-attention)
            encoder_hidden_states = hidden_states  # Key 和 Value 的来源与 Query 相同
        elif attn.norm_cross:  # 如果是交叉注意力 (cross-attention) 且需要归一化
            encoder_hidden_states = (
                attn.norm_encoder_hidden_states(  # 对 encoder_hidden_states 进行归一化
                    encoder_hidden_states
                )
            )

        key = attn.to_k(
            encoder_hidden_states
        )  # 通过线性层将 encoder_hidden_states 映射为 Key (K)
        value = attn.to_v(
            encoder_hidden_states
        )  # 通过线性层将 encoder_hidden_states 映射为 Value (V)

        query = attn.head_to_batch_dim(
            query
        )  # 为多头注意力机制重塑 Q，将 "头" 的维度合并到 "批次" 维度
        key = attn.head_to_batch_dim(key)  # 为多头注意力机制重塑 K
        value = attn.head_to_batch_dim(value)  # 为多头注意力机制重塑 V

        # 计算注意力分数 (通常是 Q 和 K 的点积，然后应用 softmax)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 将注意力分数与 V 相乘，得到加权和，即注意力的输出
        hidden_states = torch.bmm(attention_probs, value)
        # 将 "头" 和 "批次" 维度分离，恢复原始的张量结构
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj (线性投影)
        hidden_states = attn.to_out[0](hidden_states)  # 通过输出线性层进行投影
        # dropout
        hidden_states = attn.to_out[1](hidden_states)  # 应用 Dropout 防止过拟合

        if input_ndim == 4:  # 如果原始输入是4维的
            # 将张量恢复为原始的4D图像格式 [batch, channel, height, width]
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:  # 检查是否需要残差连接
            hidden_states = (
                hidden_states + residual
            )  # 将计算结果与原始输入相加（残差连接）

        hidden_states = (
            hidden_states / attn.rescale_output_factor
        )  # 对输出进行缩放，以稳定训练

        return hidden_states  # 返回最终处理后的隐藏状态
```
其中最重要的一行是：
```python
# 计算注意力分数 (通常是 Q 和 K 的点积，然后应用 softmax)
attention_probs = attn.get_attention_scores(query, key, attention_mask)
```

我们可以复制这个类，init 里加入一个存储的类，保存注意力图：
```python
class CrossAttnProcessor:  # 定义一个名为 AttnProcessor 的类
    r"""
    用于执行注意力相关计算的默认处理器。
    """

    def __init__(self, attention_store, place_in_unet):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
    ...
    def __call__(  # 定义该类的可调用方法，使其像函数一样被调用
        self,
        attn: Attention,  # 传入的 Attention 模块实例
        hidden_states,  # 输入的隐藏状态张量 (通常是图像特征)
        encoder_hidden_states=None,  # 用于交叉注意力的编码器隐藏状态，如果为 None 则是自注意力
        attention_mask=None,  # 注意力掩码，用于忽略某些位置的计算
        temb=None,  # 时间步嵌入 (Time Embedding)，通常用于条件扩散模型
    ):
    ...

        # 计算注意力分数 (通常是 Q 和 K 的点积，然后应用 softmax)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        self.attnstore(attention_probs, is_cross=True, place_in_unet=self.place_in_unet)
    ...
```

存储类的代码：
```python
class AttentionStore:
    """一个简单的注意力图存储类"""

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, **kwargs):
        # attn.shape: [batch_size * num_heads, seq_len_q, seq_len_k]
        if is_cross:
            key = f"{place_in_unet}_cross"
            # 为了演示，我们只存储注意力图
            self.step_store[key].append(attn.detach().cpu())

    def __init__(self):
        self.step_store = self.get_empty_store()
        self.attention_store = {}
```

这样就会存下一个维度为 `[B, 4096, 77]` 的注意力图。

这个注意图的含义就是每个 token 都有一个 4096 维的向量。如果转为正方形，那么就能得到 64x64 的注意力图像。

diffusers 提供了一个方便的功能，unet.attn_processors，可以用名字直接找到所有attn processor。

例如：
- 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor'
- 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor'
- 'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor'
- 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor'
- ...

命名规律是：
- block 位置
  - down_blocks.x，x 是 0, 1, 2
  - up_blocks.x，x 是 1, 2, 3
  - mid_block
- block 类型
  - attentions.x，x 是 0, 1
  - transformer_blocks.x，x 是 0
- transformer_blocks.0 类型
  - attn1.processor
  - attn2.processor

补充：
1.  **成对出现**: `attn1` (自注意力) 和 `attn2` (交叉注意力) 总是成对出现，它们共同构成一个 `BasicTransformerBlock`。
2.  **多层级注入**: 文本信息**不是只注入一次**，而是在 UNet 的**多个不同分辨率层级**（64x64, 32x32, 16x16, 8x8）中被反复注入。这使得模型能够学习到文本概念在不同尺度下的表现（例如，“猫”的全局轮廓和局部毛发纹理）。
3.  **模块数量**:
    *   `down_blocks` 的 0, 1, 2 层各有 2 个 `attentions` 块，每个块有1对 attn1/attn2，所以 `3 * 2 * 2 = 12` 个处理器。
    *   `up_blocks` 的 1, 2, 3 层各有 3 个 `attentions` 块，每个块有1对 attn1/attn2，所以 `3 * 3 * 2 = 18` 个处理器。
    *   `mid_block` 有 1 个 `attentions` 块，有1对 attn1/attn2，所以 `1 * 1 * 2 = 2` 个处理器。
    *   **总计**: `12 + 18 + 2 = 32` 个处理器

要替换 unet 里的注意力处理器，只需要把 unet.attn_processors 里的处理器替换为自定义的处理器。

```python
attn_procs = {}
for name in unet.attn_processors.keys():
    # 判断模块在UNet中的位置
    if name.startswith("mid_block"):
        place_in_unet = "mid"
    elif name.startswith("up_blocks"):
        place_in_unet = "up"
    elif name.startswith("down_blocks"):
        place_in_unet = "down"
    else:
        continue

    # 只替换交叉注意力模块 (attn2)
    if "attn2" in name:
        print(f"  - Replacing processor for: {name}")
        attn_procs[name] = CrossAttnProcessor(
            attention_store=attention_store, place_in_unet=place_in_unet
        )
    else:
        # 其他模块（如自注意力attn1）保持默认处理器
        attn_procs[name] = AttnProcessor()

# c. 执行替换
print("\nSetting new attention processors on UNet...")
unet.set_attn_processor(attn_procs)
```

这样的话，一次向前传播 unet，就会生成16个attention map：
  - Stored 6 attention maps for 'down_cross':
    - Shape of the 0 map: torch.Size([8, 4096, 77])
    - Shape of the 1 map: torch.Size([8, 4096, 77])
    - Shape of the 2 map: torch.Size([8, 1024, 77])
    - Shape of the 3 map: torch.Size([8, 1024, 77])
    - Shape of the 4 map: torch.Size([8, 256, 77])
    - Shape of the 5 map: torch.Size([8, 256, 77])
  - Stored 1 attention maps for 'mid_cross':
    - Shape of the 0 map: torch.Size([8, 64, 77])
  - Stored 9 attention maps for 'up_cross':
    - Shape of the 0 map: torch.Size([8, 256, 77])
    - Shape of the 1 map: torch.Size([8, 256, 77])
    - Shape of the 2 map: torch.Size([8, 256, 77])
    - Shape of the 3 map: torch.Size([8, 1024, 77])
    - Shape of the 4 map: torch.Size([8, 1024, 77])
    - Shape of the 5 map: torch.Size([8, 1024, 77])
    - Shape of the 6 map: torch.Size([8, 4096, 77])
    - Shape of the 7 map: torch.Size([8, 4096, 77])
    - Shape of the 8 map: torch.Size([8, 4096, 77])

这里是batch size是8，是因为8个head，因为计算attention map的时候，还是在单个head上算的。