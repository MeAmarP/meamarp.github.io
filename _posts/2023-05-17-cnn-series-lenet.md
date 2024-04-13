---
layout: post
title: "CNN Series: LeNet - Gradient-Based Learning Applied to Document Recognition"
author: "Amar P"
categories: journal
tags: [cnn-series,image-models]
---

## Paper

- **Paper Name**: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- **Authors**: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner

## LeNet:
- LeNet-5 is a pioneering convolutional neural network (CNN) that played a foundational role in the field of deep learning, particularly in the application of using deep learning to image recognition tasks.
- Key Contributions:
  - Conv Layer: Demonstrated the effectiveness of using learnable filters for feature extraction from images.
  - Subsampling/Pooling Layer: Introduced the concept of spatial pooling (also known down-sampling) to reduce the spatial size of the representation, control overfitting, and reduce computational requirements.
  - One more aspect about LeNet-5 regarding activations is that,
    - for intermediate layers it used tanh
    - for final output layer radial basis function (RBF) network to classify images.
- Architecture: LeNet-5's architecture is relatively simple by today's standards but was revolutionary at the time.

<img src="{{site.url}}/assets/img/lenet.jpg" style="display: block; margin: auto;" />
