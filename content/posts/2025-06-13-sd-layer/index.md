---
title: "Understand UNet in SD"
date: 2025-06-13T20:14:03+08:00
draft: false
tags: ['Diffusion','UNet', 'featured']
math: true
---

Stable Diffusion 里的 UNet 主要有三个输入，分别是：

| Input            | Dimensions       |
| :--------------- | :--------------- |
| 带噪的潜空间图像 | `(1, 4, 64, 64)` |
| 时间步           | `(1,)`           |
| 文本提示编码     | `(1, 77, 768)`   |

这三个输入经过 UNet 后，会得到一个输出，维度是 `(1, 4, 64, 64)`。

本文的目标就是：**解释这个输出具体是怎么得到的。**

### 整体结构

UNet 的结构主要包含五个部分：
- conv_in
- Encoder
- Middle Block
- Decoder
- conv_out

这里的 conv_in 和 conv_out 其实就是一个普通的 3x3 卷积层，用来转换通道数。

写成代码就是一行的事：

```python
conv_in = Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
conv_out = Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
```
所以 UNet 的大部份操作，其实都是在 `(1, 320, 64, 64)` 这个维度基础上完成的。

| Component    | Input Dimensions   | Output Dimensions  |
| :----------- | :----------------- | :----------------- |
| Encoder      | `(1, 320, 64, 64)` | `(1, 1280, 8, 8)`  |
| Middle Block | `(1, 1280, 8, 8)`  | `(1, 1280, 8, 8)`  |
| Decoder      | `(1, 1280, 8, 8)`  | `(1, 320, 64, 64)` |

### 先分析 Encoder

输入 `(1, 320, 64, 64)` 进入Encoder后，会依次经过四个Block：
1. CrossAttnDownBlock2D
2. CrossAttnDownBlock2D
3. CrossAttnDownBlock2D
4. DownBlock2D

得到输出 `(1, 1280, 8, 8)`。

可以看到，这里有两类 Block：
- CrossAttnDownBlock2D
- DownBlock2D

他俩的关系是：CrossAttnDownBlock2D 是加强版的 DownBlock2D，里面有 Attention Block ，而 DownBlock2D 里没有。

下面先介绍 DownBlock2D。

DownBlock2D 包含 N个ResBlock（N=2）和一个可选的下采样层。

特征会依次经过这N个ResBlock，最后经过下采样层（如果有）。

而 CrossAttnDownBlock2D 在每个ResBlock后，加了一个 Attention Block，用来计算图像特征的自注意力和图文特征的交叉注意力。

特征会依次经过这N对ResBlock-Attention Block，最后经过下采样层（如果有）。

对于Encoder里下采样层的设计是，前三个Block有，最后一个没有。

对于特征通道数的加倍是，中间两个会加倍，其他的不变。


下面是整体的设计：

| 对比指标 | Down Block 1 | Down Block 2 | Down Block 3 | Down Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `CrossAttnDownBlock2D` | `DownBlock2D` |
| **尺寸减半** | 是 | 是 | 是 | 否 |
| **尺寸变化** | 64x64 -> 32x32 | 32x32 -> 16x16 | 16x16 -> 8x8 | 8x8|
| **通道数加倍** | 否 | 是 | 是 | 否 |
| **通道数变化** | 320 | 320 -> 640 | 640 -> 1280 | 1280  |
| **ResBlock 数量** | 2 | 2 | 2 | 2 |
| **Attention Block 数量** | 2 | 2 | 2 | **0** |
| **主要输入** | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li> |



conv_in、Encoder里的每个ResBlock以及下采样层的输出，都会存入down_block_res_samples，用于后续Decoder都跨层特征融合。

所以down_block_res_samples里一共有：`1+2*4+1*3=1+8+3=12` 个特征。

这12个特征的维度如下图：

| 序号 (Index) | 生产者 (Producer Layer)           | 输出特征图 (通道数, 尺寸) |
| :----------- | :-------------------------------- | :------------------------ |
| **0**        | `conv_in`                         | (320, 64x64)              |
| **1**        | `down_blocks[0]` 的第1个 ResBlock | (320, 64x64)              |
| **2**        | `down_blocks[0]` 的第2个 ResBlock | (320, 64x64)              |
| **3**        | `down_blocks[0]` 的 Downsampler   | (320, 32x32)              |
| **4**        | `down_blocks[1]` 的第1个 ResBlock | (640, 32x32)              |
| **5**        | `down_blocks[1]` 的第2个 ResBlock | (640, 32x32)              |
| **6**        | `down_blocks[1]` 的 Downsampler   | (640, 16x16)              |
| **7**        | `down_blocks[2]` 的第1个 ResBlock | (1280, 16x16)             |
| **8**        | `down_blocks[2]` 的第2个 ResBlock | (1280, 16x16)             |
| **9**        | `down_blocks[2]` 的 Downsampler   | (1280, 8x8)               |
| **10**       | `down_blocks[3]` 的第1个 ResBlock | (1280, 8x8)               |
| **11**       | `down_blocks[3]` 的第2个 ResBlock | (1280, 8x8)               |

### 再分析 Decoder

Encdoer 的输出会经过 Middle Block，并且特征维度和通道数都保持不变。

*   **输入 `x`**: `(1, 1280, 8, 8)`
*   **内部层**: 1 x **ResBlock**, 1 x **Attention Block**, 1 x **ResBlock**。
*   **输出 `x`**: `(1, 1280, 8, 8)`

Middle Block 的最终输出，也就是 Decoder 的输入，就需要和 Encdoer 的输出列表不断的进行拼接融合了。

输入在 Decoder 里也会经过四个 Block：
1. UpBlock2D
2. CrossAttnUpBlock2D
3. CrossAttnUpBlock2D
4. CrossAttnUpBlock2D

Decoder的设计和Encoder差不多，但是有以下区别：

- 前3个block要做上采样
- 后两个block做通道数减半
- 每个block包含3对ResBlock- Attention Block，而不是2对。

| 对比指标 | Up Block 1 | Up Block 2 | Up Block 3 | Up Block 4 |
| :--- | :--- | :--- | :--- | :--- |
| **模块类型** | `UpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` | `CrossAttnUpBlock2D` |
| **尺寸加倍** | 是 | 是 | 是 | 否 |
| **尺寸变化** | 8x8 -> 16x16 | 16x16 -> 32x32 | 32x32 -> 64x64 | 64x64 -> 64x64 |
| **通道数减半** | 否 | 否 | 是 | 是 |
| **通道数变化** | 1280 + x -> 1280 | 1280 + x -> 1280 | 1280 + x -> 640 | 640 + x -> 320 |
| **ResBlock 数量** | 3 | 3 | 3 | 3 |
| **Attention Block 数量** | **0** | 3 | 3 | 3 |
| **主要输入** | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> | <li> (图像)</li><li> (时间)</li><li> (跨层连接)</li><li> (文本)</li> |

### 跨层分析

使用残差的时候，就是先拼接，再通过ResBlock融合特征通道。

因为前两个block对特征通道数不变，所以这三个ResBlock的输出维度是一样的。

**`up_blocks[0]`**

-   **输入 `sample_in`**: `[B, 1280, 8, 8]` (来自 `mid_block`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 1.1 | `torch.cat` | `sample_in`: `[B, 1280, 8, 8]` <br> `res[11]`: `[B, 1280, 8, 8]` |  | `[B, 2560, 8, 8]` |
| 1.2 | `up_blocks[0].ResBlocks[0]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResBlockBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.3 | `torch.cat` | `(上一步输出)`: `[B, 1280, 8, 8]` <br> `res[10]`: `[B, 1280, 8, 8]` |  | `[B, 2560, 8, 8]` |
| 1.4 | `up_blocks[0].ResBlocks[1]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResBlockBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.5 | `torch.cat` | `(上一步输出)`: `[B, 1280, 8, 8]` <br> `res[9]`: `[B, 1280, 8, 8]` |  | `[B, 2560, 8, 8]` |
| 1.6 | `up_blocks[0].ResBlocks[2]` | `(上一步输出)`: `[B, 2560, 8, 8]` | `ResBlockBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 8, 8]` |
| 1.7 | `up_blocks[0].upsamplers[0]` | `(上一步输出)`: `[B, 1280, 8, 8]` | `Upsample2D(conv: 1280 -> 1280)` | **`[B, 1280, 16, 16]`** |



这里也是，因为总体上前2个block的输出通道数是不变的，只是要根据残差的通道数来修改输入通道数。

**`up_blocks[1]`**

-   **输入 `sample_in`**: `[B, 1280, 16, 16]` (来自 `up_blocks[0]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 2.1 | `torch.cat` | `sample_in`: `[B, 1280, 16, 16]` <br> `res[8]`: `[B, 1280, 16, 16]` |  | `[B, 2560, 16, 16]` |
| 2.2 | `up_blocks[1].ResBlocks[0]` | `(上一步输出)`: `[B, 2560, 16, 16]` | `ResBlockBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.3 | `torch.cat` | `(上一步输出)`: `[B, 1280, 16, 16]` <br> `res[7]`: `[B, 1280, 16, 16]` |  | `[B, 2560, 16, 16]` |
| 2.4 | `up_blocks[1].ResBlocks[1]` | `(上一步输出)`: `[B, 2560, 16, 16]` | `ResBlockBlock2D(conv1: 2560 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.5 | `torch.cat` | `(上一步输出)`: `[B, 1280, 16, 16]` <br> `res[6]`: `[B, 640, 16, 16]` |  | `[B, 1920, 16, 16]` |
| 2.6 | `up_blocks[1].ResBlocks[2]` | `(上一步输出)`: `[B, 1920, 16, 16]` | `ResBlockBlock2D(conv1: 1920 -> 1280)` | `[B, 1280, 16, 16]` |
| 2.7 | `up_blocks[1].upsamplers[0]` | `(上一步输出)`: `[B, 1280, 16, 16]` | `Upsample2D(conv: 1280 -> 1280)` | **`[B, 1280, 32, 32]`** |



后两个Block对第一个ResBlock需要做到通道数减半。

**`up_blocks[2]`**

-   **输入 `sample_in`**: `[B, 1280, 32, 32]` (来自 `up_blocks[1]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | `torch.cat` | `sample_in`: `[B, 1280, 32, 32]` <br> `res[5]`: `[B, 640, 32, 32]` |  | `[B, 1920, 32, 32]` |
| 3.2 | `up_blocks[2].ResBlocks[0]` | `(上一步输出)`: `[B, 1920, 32, 32]` | `ResBlockBlock2D(conv1: 1920 -> 640)` | `[B, 640, 32, 32]` |
| 3.3 | `torch.cat` | `(上一步输出)`: `[B, 640, 32, 32]` <br> `res[4]`: `[B, 640, 32, 32]` | `N/A` | `[B, 1280, 32, 32]` |
| 3.4 | `up_blocks[2].ResBlocks[1]` | `(上一步输出)`: `[B, 1280, 32, 32]` | `ResBlockBlock2D(conv1: 1280 -> 640)` | `[B, 640, 32, 32]` |
| 3.5 | `torch.cat` | `(上一步输出)`: `[B, 640, 32, 32]` <br> `res[3]`: `[B, 320, 32, 32]` | `N/A` | `[B, 960, 32, 32]` |
| 3.6 | `up_blocks[2].ResBlocks[2]` | `(上一步输出)`: `[B, 960, 32, 32]` | `ResBlockBlock2D(conv1: 960 -> 640)` | `[B, 640, 32, 32]` |
| 3.7 | `up_blocks[2].upsamplers[0]` | `(上一步输出)`: `[B, 640, 32, 32]` | `Upsample2D(conv: 640 -> 640)` | **`[B, 640, 64, 64]`** |



第一个ResBlock要做到通道数减半。

**`up_blocks[3]`**

-   **输入 `sample_in`**: `[B, 640, 64, 64]` (来自 `up_blocks[2]`)

| 步骤 | 操作 | 输入Tensor(s) 及其维度 | 关键层定义 (in_ch -> out_ch) | 输出维度 |
| :--- | :--- | :--- | :--- | :--- |
| 4.1 | `torch.cat` | `sample_in`: `[B, 640, 64, 64]` <br> `res[2]`: `[B, 320, 64, 64]` |  | `[B, 960, 64, 64]` |
| 4.2 | `up_blocks[3].ResBlocks[0]` | `(上一步输出)`: `[B, 960, 64, 64]` | `ResBlockBlock2D(conv1: 960 -> 320)` | `[B, 320, 64, 64]` |
| 4.3 | `torch.cat` | `(上一步输出)`: `[B, 320, 64, 64]` <br> `res[1]`: `[B, 320, 64, 64]` |  | `[B, 640, 64, 64]` |
| 4.4 | `up_blocks[3].ResBlocks[1]` | `(上一步输出)`: `[B, 640, 64, 64]` | `ResBlockBlock2D(conv1: 640 -> 320)` | `[B, 320, 64, 64]` |
| 4.5 | `torch.cat` | `(上一步输出)`: `[B, 320, 64, 64]` <br> `res[0]`: `[B, 320, 64, 64]` | `N/A` | `[B, 640, 64, 64]` |
| 4.6 | `up_blocks[3].ResBlocks[2]` | `(上一步输出)`: `[B, 640, 64, 64]` | `ResBlockBlock2D(conv1: 640 -> 320)` | `[B, 320, 64, 64]` |