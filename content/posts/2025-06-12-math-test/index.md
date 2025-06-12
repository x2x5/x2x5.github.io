---
title: "弄个博客太麻烦了"
date: 2025-06-12T19:03:28+08:00
draft: false
math: true
tags: ["Math", "Test"]
---

本来想把平时用 markdown 写的笔记发到网上，方便用不同的电脑的时候都能看。

照理说这个事应该很简单，在知乎直接发就完事了。但我发现知乎上传markdown文件，那个公式全是乱的。

我就想那就像我网上之前看过的很多人一样，发在xx.github.io得了，应该这一套流程是很成熟了，毕竟那么多人整。

结果就这点事情，我弄了两天了，遇到一堆乱七八糟的问题。

其他抱怨的话也懒得说了，这里把最后怎么弄好的步骤记下来。


`$$
\frac{\partial E(\boldsymbol{w})}{\partial z_j} = \sum\limits_{k}\frac{\partial E(\boldsymbol{w})}{\partial y_{k}}\frac{\partial y_k}{\partial z_{j}}= \sum\limits_{k} (y_{k}- \hat{y}_{k}) w_{kj}^{(2)} \tag{5.11}
$$`
