# Deep Learning Exercises [DL E]

## Exercise 3: Regularization and the Recurrent Layer

### Introduction
This exercise is designed to extend your framework to include common regularization strategies and recurrent layers. You will implement various regularizers, dropout, batch normalization, and recurrent neural network (RNN) layers.

### What I Learned

1. **Regularizers**:
    - Implementing L1 and L2 regularization schemes.
    - Understanding the importance of regularization in preventing overfitting.
    - Refactoring optimizers to apply regularizers and adding regularization loss to the total loss.

2. **Dropout**:
    - Implementing the dropout regularization technique.
    - Understanding the concept of inverted dropout and its implementation during training and testing phases.

3. **Batch Normalization**:
    - Implementing batch normalization for both vector and image-like tensors.
    - Understanding the importance of moving average estimation of mean and variance.
    - Implementing the reformatting of tensors for batch normalization.

4. **LeNet Architecture (Optional)**:
    - Implementing a variant of the LeNet architecture.
    - Saving and loading neural networks using Python's pickle.
    - Training the LeNet variant on the MNIST dataset.

5. **Recurrent Layers**:
    - Implementing activation functions: TanH and Sigmoid.
    - Implementing Elman Recurrent Neural Network (RNN) layers.
    - Understanding the concept of memorization in RNNs and implementing the memorize property.
    - Implementing Long Short-Term Memory (LSTM) layers (optional).

### Files Implemented

- `Constraints.py`: Contains the classes `L2Regularizer` and `L1Regularizer` for regularization.
- `Dropout.py`: Contains the `Dropout` class for dropout regularization.
- `BatchNormalization.py`: Contains the `BatchNormalization` class for batch normalization.
- `LeNet.py`: Contains the implementation of the LeNet architecture.
- `TanH.py`: Contains the `TanH` class for the TanH activation function.
- `Sigmoid.py`: Contains the `Sigmoid` class for the Sigmoid activation function.
- `RNN.py`: Contains the `RNN` class for Elman Recurrent Neural Networks.
- `LSTM.py`: Contains the `LSTM` class for Long Short-Term Memory networks (optional).
- `NeuralNetworkTests.py`: Unit tests to verify the correctness of the implementations.

### Running the Code

To run the unit tests for each component:
```sh
python NeuralNetworkTests.py <TestName>