import jax.numpy as jnp
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
         # Store the actual input shape for backward pass
        self.actual_input_shape = input.shape

         # Handle batch dimension: (batch, C, H, W) -> (batch, features, 1)
        batch_size = input.shape[0]
        
        return jnp.reshape(input, (batch_size,) + self.output_shape)


    def backward(self, output_gradient, learning_rate):
        return jnp.reshape(output_gradient, self.actual_input_shape)