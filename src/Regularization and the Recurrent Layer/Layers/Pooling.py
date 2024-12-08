# import numpy as np
# import math

# class Pooling:
#     def __init__(self, stride_shape, pooling_shape):
#         self.stride_shape = stride_shape
#         self.pooling_shape = pooling_shape
#         self.input_tensor = None
#         self.trainable = False

#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         stride_y, stride_x = self.stride_shape
#         pool_y, pool_x = self.pooling_shape
#         batch_size, depth, height, width = input_tensor.shape

#         output_height = math.ceil((height - pool_y) // stride_y) + 1
#         output_width = math.ceil((width - pool_x) // stride_x) + 1

#         self.mask = np.zeros_like(input_tensor)

#         output = np.zeros((batch_size, depth, output_height, output_width))

#         for n in range(batch_size):
#             for c in range(depth):
#                 for i in range(output_height):
#                     for j in range(output_width):
#                         start_i = i * stride_y
#                         start_j = j * stride_x
#                         patch = input_tensor[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x]
#                         output[n, c, i, j] = np.max(patch)
#                         max_positions = np.where(patch == np.max(patch))
#                         for pos in zip(*max_positions):
#                             self.mask[n, c, start_i+pos[0], start_j+pos[1]] += 1
#         return output

#     def backward(self, error_tensor):
#         stride_y, stride_x = self.stride_shape
#         pool_y, pool_x = self.pooling_shape
#         batch_size, depth, _, _ = self.input_tensor.shape
#         _, _, height, width = error_tensor.shape

#         output = np.zeros_like(self.input_tensor)

#         for n in range(batch_size):
#             for c in range(depth):
#                 for i in range(height):
#                     for j in range(width):
#                         start_i = i * stride_y
#                         start_j = j * stride_x
#                         patch = self.mask[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x]
#                         output[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x] += error_tensor[n, c, i, j] * (patch > 0)

#         return output
    
    



import numpy as np
import math

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.cache = None
        self.trainable = False

    def forward(self, input_tensor):
        stride_y, stride_x = self.stride_shape
        pool_y, pool_x = self.pooling_shape
        batch_size, depth, height, width = input_tensor.shape

        output_height = math.ceil((height - pool_y) // stride_y) + 1
        output_width = math.ceil((width - pool_x) // stride_x) + 1

        self.mask = np.zeros_like(input_tensor)

        output = np.zeros((batch_size, depth, output_height, output_width))

        for n in range(batch_size):
            for c in range(depth):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride_y
                        start_j = j * stride_x
                        patch = input_tensor[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x]
                        output[n, c, i, j] = np.max(patch)
                        max_positions = np.where(patch == np.max(patch))
                        for pos in zip(*max_positions):
                            self.mask[n, c, start_i+pos[0], start_j+pos[1]] += 1

        self.cache = (input_tensor.shape, stride_y, stride_x, pool_y, pool_x)
        return output

    def backward(self, error_tensor):
        input_shape, stride_y, stride_x, pool_y, pool_x = self.cache
        batch_size, depth, _, _ = input_shape
        _, _, height, width = error_tensor.shape

        output = np.zeros_like(self.mask)

        for n in range(batch_size):
            for c in range(depth):
                for i in range(height):
                    for j in range(width):
                        start_i = i * stride_y
                        start_j = j * stride_x
                        patch = self.mask[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x]
                        output[n, c, start_i:start_i+pool_y, start_j:start_j+pool_x] += error_tensor[n, c, i, j] * (patch > 0)

        return output