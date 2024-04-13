---
layout: post
title: "CNN Series: VGGNet"
author: "Amar P"
categories: journal
tags: [cnn-series,image-models]
---

## Paper

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Authors**: Karen Simonyan, Andrew Zisserman

## VGGNet
- VGGNet are series of Convolutional Neural Network models proposed by the Visual Geometry Group of Oxford University, including the well known VGG11,VGG13,VGG16 and VGG19. 
- **Simplicity of Architecture:** VGGNet's architecture is highly uniform, utilizing 3x3 convolutional layers stacked on top of each other..
- **Increased Depth:** One of the primary innovations of VGGNet was to demonstrate that the depth of the network is a critical component for achieving high performance.
- Most of the aspects for network remains same as compared to AlexNet. It uses Max-pooling layer to reduce the spatial size of feature maps
- VGGNet uses ReLU activation throughout the network.
- VGGNet was trained using a multi-scale approach where the scale of the input images was varied during the training process. This helped the model to become robust to different image sizes and resolutions.

<img src="{{site.url}}/assets/img/vggnet.png" style="display: block; margin: auto;" />
