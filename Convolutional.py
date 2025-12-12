from layer import Layer
import jax 
import jax.numpy as jnp
from jax.scipy import signal 

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, padding=0, stride=1):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        self.depth = depth  # filter_no
        
        self.padding = padding
        self.stride = stride

        # H_out = floor((H + 2P - F) / S) + 1
        # W_out = floor((W + 2P - F) / S) + 1
        self.output_height = int(jnp.floor((input_height + 2 * self.padding - self.kernel_size) / self.stride) + 1)
        self.output_width = int(jnp.floor((input_width + 2 * self.padding - self.kernel_size) / self.stride) + 1)

        self.output_shape = (depth, self.output_height, self.output_width)

        # weights(kernels), shape=(filter_no, input_depth, kernel_size, kernel_size)
        self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)

        seed = 0
        key = jax.random.PRNGKey(seed)
        wkey, bkey = jax.random.split(key)

        self.weights = jax.random.normal(wkey, self.kernel_shape) 
        self.biases = jnp.zeros(self.output_shape)
    def forward(self, input):
        self.input = input
        batch_size = input.shape[0]
        
        # Create output with batch dimension: (batch, depth, height, width)
        self.output = jnp.zeros((batch_size,) + self.output_shape)

        # Loop over batch
        for b in range(batch_size):
            for i in range(self.depth):  # filter i
                for j in range(self.input_depth):  # depth channel
                    
                    # Calculate cross-correlation on the b-th sample
                    # input[b, j] gives us a 2D array (H, W)
                    correlation = signal.correlate2d(self.input[b, j], self.weights[i, j], mode='valid')
                    
                    # Add to the i-th output map for batch b
                    self.output = self.output.at[b, i].add(correlation)
                
                # Add bias to the i-th output map for batch b
                self.output = self.output.at[b, i].add(self.biases[i])
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        batch_size = output_gradient.shape[0]
        
        kernels_gradient = jnp.zeros(self.kernel_shape)
        input_gradient = jnp.zeros(self.input.shape)

        # Loop over batch
        for b in range(batch_size):
            for i in range(self.depth):  # Loop through each Filter (i)
                for j in range(self.input_depth):  # each input channel/depth
                
                    # 1. Calculate Kernel Gradient (dK)
                    # ∂E/∂Kij = Xj ⋆ ∂E/∂Yi
                    correlation = signal.correlate2d(self.input[b, j], output_gradient[b, i], "valid")
                    kernels_gradient = kernels_gradient.at[i, j].add(correlation)

                    # 2. Calculate Input Gradient (dX)
                    # ∂E/∂X = ∂E/∂Y ∗ full K
                    convolution = signal.convolve2d(output_gradient[b, i], self.weights[i, j], "full")
                    input_gradient = input_gradient.at[b, j].add(convolution)

        # Average gradients over batch
        kernels_gradient = kernels_gradient / batch_size
        
        # Update Weights
        self.weights = self.weights - learning_rate * kernels_gradient
        
        # Update Biases (sum over batch, then average)
        bias_gradient = jnp.sum(output_gradient, axis=0) / batch_size
        self.biases = self.biases - learning_rate * bias_gradient

        return input_gradient