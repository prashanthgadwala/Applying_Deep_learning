import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        exps = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
    
    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        self.jacobian_matrix = np.zeros((self.output.shape[0], self.output.shape[1], self.output.shape[1]))
        
        for i in range(self.jacobian_matrix.shape[0]):
            for j in range(self.jacobian_matrix.shape[1]):
                for k in range(self.jacobian_matrix.shape[2]):
                    if j == k:
                        self.jacobian_matrix[i, j, k] = self.output[i, j] * (1 - self.output[i, j])
                    else:
                        self.jacobian_matrix[i, j, k] = -self.output[i, j] * self.output[i, k]
        
        error = np.einsum('ijk,ik->ij', self.jacobian_matrix, self.error_tensor)
        return error