---
layout: post
title: "CNN Series: ResNet"
author: "Amar P"
categories: journal
tags: [cnn-series,image-models]
---

## Paper

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

## ResNet
- ResNet short for **Residual Network**

- it tackled a major challenge faced by deep convolutional neural networks (CNNs) callws **vanishing gradients**.

- As CNNs grew deeper (more layers), training became increasingly difficult. The gradients used to update the network's weights during backprop would become very small or vanish entirely in the earlier layers. This hampered learning and prevented the network from effectively utilizing its depth.

- **Residual Learning:** By adding the original input to the output of the stacked layers within a residual block, ResNet essentially allows the network to learn the residual function between the input and the desired output. This addresses the degradation problem by using this shortcut connections to skip one or more layers.
- **Identity Mapping:** Shortcut connections perform identity mapping, allowing information to bypass certain layers.
- **Skip Connections:** These connections bypass a portion of the network, **allowing the input from a lower layer to be directly added to the output of a higher layer**. This ensures that information can flow through the network unobstructed, even in very deep architectures.

- **Residual Blocks**: These blocks are the core building blocks of the ResNet architecture and contain a critical elements.

- There are two types of Residual blocks called as **BasicBlock** and **BottleNeckBlock**.

- Differences Between **BasicBlock** and **BottleneckBlock**:
  - **Depth of Layers:**
    - BasicBlock: Consists of two convolutional layers (3x3).
    - BottleneckBlock: Consists of three convolutional layers (1x1, 3x3, 1x1). 
  - **Use in Architectures:**
    - BasicBlock: Typically used in shallower networks (e.g., ResNet-18, ResNet-34).
    - BottleneckBlock: Typically used in deeper networks (e.g., ResNet-50, ResNet-101, ResNet-152).

- Common variants include ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152.

- ResNet won the 1st place in the ILSVRC 2015 classification task.

<img src="{{site.url}}/assets/img/residual-learning.png" style="display: block; margin: auto;" />
<img src="{{site.url}}/assets/img/resnet-sample.png" style="display: block; margin: auto;" />
<img src="{{site.url}}/assets/img/resnet-blocks.png" style="display: block; margin: auto;" />
<img src="{{site.url}}/assets/img/resnet-arch.png" style="display: block; margin: auto;" />
<img src="{{site.url}}/assets/img/resnet-train-error.png" style="display: block; margin: auto;" />
