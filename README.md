# Deep Learning Exercises [DL E]

## Overview

This repository contains a series of exercises designed to introduce and deepen your understanding of various deep learning concepts and techniques. The exercises cover topics such as neural networks, convolutional neural networks (CNNs), regularization, recurrent neural networks (RNNs), and using PyTorch for classification tasks.

## Table of Contents

1. [Exercise 0: Numpy Tutorial](#exercise-0-numpy-tutorial)
2. [Exercise 1: Neural Networks](#exercise-1-neural-networks)
3. [Exercise 2: Convolutional Neural Networks](#exercise-2-convolutional-neural-networks)
4. [Exercise 3: Regularization and the Recurrent Layer](#exercise-3-regularization-and-the-recurrent-layer)
5. [Exercise 4: PyTorch for Classification](#exercise-4-pytorch-for-classification)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Exercise 0: Numpy Tutorial

### Introduction
This exercise is designed to refresh your knowledge of Python and NumPy. You will use NumPy functions to create different types of patterns and implement an image generator class to load and synthetically augment data using rigid transformations.

### What I Learned
- Array manipulation with NumPy
- Creating and visualizing patterns
- Implementing an image generator class

## Exercise 1: Neural Networks

### Introduction
This exercise introduces the basics of neural networks and their implementation. You will implement various components of a neural network, including layers, activation functions, loss functions, and an optimizer.

### What I Learned
- Implementing the Stochastic Gradient Descent (SGD) optimizer
- Creating a base layer class
- Implementing fully connected layers, ReLU, SoftMax, and cross-entropy loss

## Exercise 2: Convolutional Neural Networks

### Introduction
This exercise extends the framework to include the building blocks for modern CNNs. You will implement initialization schemes, advanced optimizers, convolutional layers, max-pooling layers, and a flatten layer.

### What I Learned
- Implementing various initialization schemes
- Advanced optimization schemes: SgdWithMomentum and Adam
- Implementing convolutional and pooling layers

## Exercise 3: Regularization and the Recurrent Layer

### Introduction
This exercise introduces common regularization strategies and recurrent layers. You will implement various regularizers, dropout, batch normalization, and RNN layers.

### What I Learned
- Implementing L1 and L2 regularization schemes
- Dropout and batch normalization
- Implementing Elman Recurrent Neural Network (RNN) layers

## Exercise 4: PyTorch for Classification

### Introduction
This exercise involves using PyTorch to implement a version of the ResNet architecture and the necessary workflow surrounding deep learning algorithms. The task is to detect defects on solar cells using a provided dataset.

### What I Learned
- Implementing a dataset container in PyTorch
- Building and training a ResNet model
- Using PyTorch for data loading, preprocessing, and augmentation

## Installation

To set up the environment, use the provided `environment.yml` file:
```sh
conda env create -f environment.yml
conda activate deep-learning-exercises
```

## Usage

### Running Unit Tests
To run the unit tests for each component:
```sh
python NeuralNetworkTests.py <TestName>
```

### To run all unit tests:
```sh
python NeuralNetworkTests.py
```

### To run the bonus tests:
```sh
python3 NeuralNetworkTests.py Bonus
```

### Training Models

- Data Preparation:
    - Ensure the dataset is in the correct directory.
    - Update the paths in the configuration files if necessary.

### Training:
```sh
python train.py
```

### Evaluation:
```sh
python evaluate.py
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.