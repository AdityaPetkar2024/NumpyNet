# NumpyNet

A neural network library implemented from scratch using only NumPy — 
no PyTorch, TensorFlow, or autograd. Built while working through 
*Neural Networks from Scratch* (Kinsley & Kukiela), implementing every 
forward pass, backward pass, and parameter update manually.

## What's implemented

- Dense layers with L1/L2 regularization
- ReLU and Softmax activations
- Categorical Cross-Entropy loss, with a combined Softmax+CrossEntropy 
  backward pass (~7x faster than computing separately)
- Dropout (inverted dropout, training/inference modes)
- Adam optimizer (momentum + adaptive learning rates + bias correction)
- Mini-batch training with accumulated loss/accuracy across epochs
- A `Model` class that chains layers via `.prev`/`.next` references for 
  uniform forward/backward loops

## Result

Trained on Fashion MNIST (60,000 train / 10,000 test images, 28x28 
grayscale, 10 classes):

- **87.9% validation accuracy** after 10 epochs
- Training and validation accuracy stay close (88.6% vs 87.9%) — L2 + 
  dropout are preventing significant overfitting

## What I found interesting

The confusion matrix shows the model struggles most with **Shirt (65% 
accuracy)** — confused with T-shirt/top, Coat, and Pullover. These are 
all upper-body garments with overlapping silhouettes. A flattened 
784-pixel vector loses the spatial detail (collar shape, sleeve length) 
that would distinguish them — a known limitation of dense networks on 
image data, and exactly the problem convolutional layers are designed 
to solve.

## Notebook

[`numpynet_fashion_mnist.ipynb`](./numpynet_fashion_mnist.ipynb) — full 
implementation, training run, confusion matrix, and analysis.

## Next

Implementing `Layer_Conv2D` and `Layer_MaxPool2D` from scratch to test 
whether spatial structure improves the Shirt/Coat/Pullover confusion.
