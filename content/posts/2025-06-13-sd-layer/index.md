---
title: "Stable Diffusion UNet 结构分析"
date: 2025-06-13T20:14:03+08:00
draft: false
tags: ['Diffusion','UNet', 'featured']
math: true
---

**从输入到输出：**

Stable Diffusion UNet 接收三个输入，分别是：
- 带噪点的潜空间图像，维度是 `(1, 4, 64, 64)`
- 时间步，维度是 `(1,)`
- 文本提示编码，维度是 `(1, 77, 768)`

这三个输入经过 UNet 后，会得到一个输出，维度是 `(1, 4, 64, 64)`。

**结构分析：**

UNet 的结构可以分为五个部分：
- Initial Conv
- Encoder
- Middle Block
- Decoder
- Final Conv

先分析简单的 Initial Conv 和 Final Conv。

这两个东西，其实就是一个普通的卷积层，作用分别是把输入的维度从 `(1, 4, 64, 64)` 变成了 `(1, 320, 64, 64)`，和把输出从 `(1, 320, 64, 64)` 变成了 `(1, 4, 64, 64)`。写成代码都是一行的事：

```python
# Initial Conv
Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# Final Conv
Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```
所以 UNet 主要的部分，都是在 `(1, 320, 64, 64)` 维度上操作的。

从总体上看，这两个部分的维度变化分别是：
- Encoder 的维度变化：`(1, 320, 64, 64)` -> `(1, 1280, 8, 8)`
- Middle Block 的维度变化：`(1, 1280, 8, 8)` -> `(1, 1280, 8, 8)`
- Decoder 的维度变化：`(1, 1280, 8, 8)` -> `(1, 320, 64, 64)`

下面先重点分析 Encoder，另外两个的设计和Encoder差不多。

### 结构分析
#### Encoder 结构分析

输入进入Encoder后，会依次经过四个Block：
1. CrossAttnDownBlock2D
2. CrossAttnDownBlock2D
3. CrossAttnDownBlock2D
4. DownBlock2D

可以看到，这四个Block分为两类：
- CrossAttnDownBlock2D
- DownBlock2D


这两类 Block 的共同点是：他俩都包含2个resnet，都有一个可选的下采样层。

而 CrossAttnDownBlock2D 还包含 2个transformer，可以计算自注意力和交叉注意力。


下面通过表格对比这四个Block：

| 对比指标 | Down Block 1 | Down Block 2 | Down Block 3 | Down Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `DownBlock2D` |
| **尺寸减半** | 是 | 是 | 是 | 否 |
| **尺寸变化** | 64x64 -> 32x32 | 32x32 -> 16x16 | 16x16 -> 8x8 | 8x8|
| **通道数加倍** | 否 | 是 | 是 | 否 |
| **通道数变化** | 320 | 320 -> 640 | 640 -> 1280 | 1280  |
| **ResNet 数量** | 2 | 2 | 2 | 2 |
| **Transformer 数量**| 2 | 2 | 2 | **0** |
| **主要输入** | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li> |


通过以上分析，可以看到，ResBlock 和 Attention Block，下采样层的组合，构成了 Encoder。

而剩下的 Middle Block 和 Decoder 的设计也是类似的。

下面具体分析。

#### Middle Block 结构分析

Middle Block 包含 2个 ResNet 和 1个Transformer，没有下采样层，输出维度也和输入一样。

*   **输入 `x`**: `(1, 1280, 8, 8)`
*   **内部层**: 1 x **ResBlock**, 1 x **Attention Block**, 1 x **ResBlock**。
*   **输出 `x`**: `(1, 1280, 8, 8)`

#### Decoder 结构分析

Decoder 需要考虑和 Encoder 对称，并且融合跨层连接。

输入在 Decoder 里也会经过四个 Block：
1. UpBlock2D
2. CrossAttnUpBlock2D
3. CrossAttnUpBlock2D
4. CrossAttnUpBlock2D

在 Stable Diffusion UNet 里，Encoder 和 Decoder 的设计都是包含4个block，并且前3个block来做下采样或者上采样。

Decoder 里的每个 Block，包含3对 resnet 和 transformer，而 Encoder 里是2对。

| 对比指标 | Up Block 1 | Up Block 2 | Up Block 3 | Up Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `UpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` |
| **尺寸加倍** | 是 | 是 | 是 | 否 |
| **尺寸变化** | 8x8 -> 16x16 | 16x16 -> 32x32 | 32x32 -> 64x64 | 64x64 -> 64x64 |
| **通道数减半** | 否 | 否 | 是 | 是 |
| **通道数变化** | 1280 + x -> 1280 | 1280 + x -> 1280 | 1280 + x -> 640 | 640 + x -> 320 |
| **ResNet 数量** | 3 | 3 | 3 | 3 |
| **Transformer 数量**| **0** | 3 | 3 | 3 |
| **主要输入** | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> |


### 残差分析

#### 产生残差

`down_block_res_samples` 这个 `tuple` 像一个**栈 (Stack)**，编码器按顺序把残差**压入 (Push)**，解码器则从末尾**弹出 (Pop)** 使用。这种 **后进先出 (LIFO)** 的机制确保了对称的连接。

下表详细列出了每一个残差的来源和去向。

| 序号 (Index) | 生产者 (Producer Layer) | 输出特征图 (通道数, 尺寸) | 
| :--- | :--- | :--- | 
| **0** | `conv_in` | (320, 64x64) | 
| **1** | `down_blocks[0]` 的第1个 ResNet | (320, 64x64) |   
| **2** | `down_blocks[0]` 的第2个 ResNet | (320, 64x64) |   
| **3** | `down_blocks[0]` 的 Downsampler | (320, 32x32) |   
| **4** | `down_blocks[1]` 的第1个 ResNet | (640, 32x32) | 
| **5** | `down_blocks[1]` 的第2个 ResNet | (640, 32x32) |  
| **6** | `down_blocks[1]` 的 Downsampler | (640, 16x16) |  
| **7** | `down_blocks[2]` 的第1个 ResNet | (1280, 16x16) |
| **8** | `down_blocks[2]` 的第2个 ResNet | (1280, 16x16) |
| **9** | `down_blocks[2]` 的 Downsampler | (1280, 8x8) | 
| **10** | `down_blocks[3]` 的第1个 ResNet | (1280, 8x8) |  
| **11** | `down_blocks[3]` 的第2个 ResNet | (1280, 8x8) | 

**这些残差是怎么用的？**


#### 使用残差


##### **`up_blocks[0]`**
-   **输入 `sample_in`**: `[B, 1280, 8, 8]` (来自 `mid_block`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | `torch.cat` | `sample_in`: `[B, 1280, 8, 8]` <br> `res[11]`: `[B, 1280, 8, 8]` | `N/A` | `[B, 2560, 8, 8]` |
| 1.2 | `up_blocks[0].resnets[0]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResnetBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.3 | `torch.cat` | `(上一步输出)`: `[B, 1280, 8, 8]` <br> `res[10]`: `[B, 1280, 8, 8]` | `N/A` | `[B, 2560, 8, 8]` |
| 1.4 | `up_blocks[0].resnets[1]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResnetBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.5 | `torch.cat` | `(上一步输出)`: `[B, 1280, 8, 8]` <br> `res[9]`: `[B, 1280, 8, 8]` | `N/A` | `[B, 2560, 8, 8]` |
| 1.6 | `up_blocks[0].resnets[2]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResnetBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.7 | `up_blocks[0].upsamplers[0]` | `(上一步输出)`: `[B, 1280, 8, 8]` | `Upsample2D(conv: 1280 -> 1280)` | **`[B, 1280, 16, 16]`** |

##### **`up_blocks[1]`**
-   **输入 `sample_in`**: `[B, 1280, 16, 16]` (来自 `up_blocks[0]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | `torch.cat` | `sample_in`: `[B, 1280, 16, 16]` <br> `res[8]`: `[B, 1280, 16, 16]` | `N/A` | `[B, 2560, 16, 16]` |
| 2.2 | `up_blocks[1].resnets[0]` | `(上一步输出)`: `[B, 2560, 16, 16]` | `ResnetBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.3 | `torch.cat` | `(上一步输出)`: `[B, 1280, 16, 16]` <br> `res[7]`: `[B, 1280, 16, 16]` | `N/A` | `[B, 2560, 16, 16]` |
| 2.4 | `up_blocks[1].resnets[1]` | `(上一步输出)`: `[B, 2560, 16, 16]` | `ResnetBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.5 | `torch.cat` | `(上一步输出)`: `[B, 1280, 16, 16]` <br> `res[6]`: `[B, 640, 16, 16]` | `N/A` | `[B, 1920, 16, 16]` |
| 2.6 | `up_blocks[1].resnets[2]` | `(上一步输出)`: `[B, 1920, 16, 16]` | `ResnetBlock2D(conv1: 1920 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.7 | `up_blocks[1].upsamplers[0]` | `(上一步输出)`: `[B, 1280, 16, 16]` | `Upsample2D(conv: 1280 -> 1280)` | **`[B, 1280, 32, 32]`** |

##### **`up_blocks[2]`**
-   **输入 `sample_in`**: `[B, 1280, 32, 32]` (来自 `up_blocks[1]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | `torch.cat` | `sample_in`: `[B, 1280, 32, 32]` <br> `res[5]`: `[B, 640, 32, 32]` | `N/A` | `[B, 1920, 32, 32]` |
| 3.2 | `up_blocks[2].resnets[0]` | `(上一步输出)`: `[B, 1920, 32, 32]` | `ResnetBlock2D(conv1: 1920 -> 640)` | `[B, 640, 32, 32]` |
| 3.3 | `torch.cat` | `(上一步输出)`: `[B, 640, 32, 32]` <br> `res[4]`: `[B, 640, 32, 32]` | `N/A` | `[B, 1280, 32, 32]` |
| 3.4 | `up_blocks[2].resnets[1]` | `(上一步输出)`: `[B, 1280, 32, 32]` | `ResnetBlock2D(conv1: 1280 -> 640)` | `[B, 640, 32, 32]` |
| 3.5 | `torch.cat` | `(上一步输出)`: `[B, 640, 32, 32]` <br> `res[3]`: `[B, 320, 32, 32]` | `N/A` | `[B, 960, 32, 32]` |
| 3.6 | `up_blocks[2].resnets[2]` | `(上一步输出)`: `[B, 960, 32, 32]` | `ResnetBlock2D(conv1: 960 -> 640)` | `[B, 640, 32, 32]` |
| 3.7 | `up_blocks[2].upsamplers[0]` | `(上一步输出)`: `[B, 640, 32, 32]` | `Upsample2D(conv: 640 -> 640)` | **`[B, 640, 64, 64]`** |

##### **`up_blocks[3]`**
-   **输入 `sample_in`**: `[B, 640, 64, 64]` (来自 `up_blocks[2]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | `torch.cat` | `sample_in`: `[B, 640, 64, 64]` <br> `res[2]`: `[B, 320, 64, 64]` | `N/A` | `[B, 960, 64, 64]` |
| 4.2 | `up_blocks[3].resnets[0]` | `(上一步输出)`: `[B, 960, 64, 64]` | `ResnetBlock2D(conv1: 960 -> 320)` | `[B, 320, 64, 64]` |
| 4.3 | `torch.cat` | `(上一步输出)`: `[B, 320, 64, 64]` <br> `res[1]`: `[B, 320, 64, 64]` | `N/A` | `[B, 640, 64, 64]` |
| 4.4 | `up_blocks[3].resnets[1]` | `(上一步输出)`: `[B, 640, 64, 64]` | `ResnetBlock2D(conv1: 640 -> 320)` | `[B, 320, 64, 64]` |
| 4.5 | `torch.cat` | `(上一步输出)`: `[B, 320, 64, 64]` <br> `res[0]`: `[B, 320, 64, 64]` | `N/A` | `[B, 640, 64, 64]` |
| 4.6 | `up_blocks[3].resnets[2]` | `(上一步输出)`: `[B, 640, 64, 64]` | `ResnetBlock2D(conv1: 640 -> 320)` | `[B, 320, 64, 64]` |