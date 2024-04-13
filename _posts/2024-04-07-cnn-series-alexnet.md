---
layout: post
title: "CNN Series: LeNet - Gradient-Based Learning Applied to Document Recognition"
author: "Amar P"
categories: journal
tags: [cnn-series,image-models]
---

## Paper

- **Paper Name**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

## AlexNet
- AlexNet is a convolutional neural network that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It was designed to recognize objects and faces from over a million different categories, making it one of the most successful applications of deep learning.
- **ReLU** as its activation function was used, which addresses issue of vanishing gradient problem that is common with sigmoid or tanh.
- **DropOut** was used to randomly ignore neurons during training to reduce overfitting.
- In Convloutional Layers of Alexnet, **overlapping max pooling**, as sub-sampling layer, was used to reduce the spatial size (width and height) of the representation at each stage.
- **Deep Architecture:** It has 8 layers in total: 5 convolutional layers and 3 fully connected layers.
- Designed to **utilize GPUs**, which significantly sped up the training process.
- AlexNet used **data augmentation** techniques such as image translations, horizontal reflections, and patch extractions to combat overfitting.

<img src="{{site.url}}/assets/img/alexnet.jpg" style="display: block; margin: auto;" />
