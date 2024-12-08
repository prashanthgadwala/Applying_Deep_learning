# Deep Learning Exercises [DL E]

## Exercise 0: Numpy Tutorial

### Introduction
This exercise is designed to refresh your knowledge of Python and NumPy. You will use NumPy functions to create different types of patterns and implement an image generator class to load and synthetically augment data using rigid transformations.

### What I Learned

1. **Array Manipulation with NumPy**:
    - Implementing patterns using NumPy functions without loops.
    - Proper NumPy array indexing and slicing.
    - Creating and visualizing patterns using `matplotlib`.

2. **Checkerboard Pattern**:
    - Creating a checkerboard pattern with adaptable tile size and resolution.
    - Ensuring the resolution is divisible by the tile size without remainder.
    - Visualizing the pattern using `plt.imshow()`.

3. **Binary Circle Pattern**:
    - Drawing a binary circle with a given radius at a specified position using NumPy operations.
    - Understanding the formula describing the circle with respect to pixel coordinates.
    - Using `np.meshgrid` for pattern creation.

4. **RGB Color Spectrum**:
    - Creating an RGB color spectrum with rising values across specific dimensions.
    - Handling RGB images with 3 channels and ensuring intensity values range from 0.0 to 1.0.

5. **Data Handling and Augmentation**:
    - Implementing an `ImageGenerator` class to read images and their associated class labels from a JSON file.
    - Generating batches of data for training neural networks.
    - Implementing data augmentation techniques such as shuffling, mirroring, and rotation.
    - Ensuring all batches have the same size and handling resizing of images.

6. **Unit Testing**:
    - Running unit tests to verify the correctness of implementations.
    - Debugging implementations until all tests pass.
    - Using IDEs like PyCharm for running and managing unit tests.

### Files Implemented

- `pattern.py`: Contains the classes `Checker`, `Circle`, and `Spectrum` for creating different patterns.
- `main.py`: Imports and calls the classes from `pattern.py` for debugging and visualization.
- `generator.py`: Contains the `ImageGenerator` class for data handling and augmentation.
- `NumpyTests.py`: Unit tests to verify the correctness of the implementations.

### Running the Code

To run all unit tests:
```sh
python NumpyTests.py