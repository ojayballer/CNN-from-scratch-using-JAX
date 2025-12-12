from layer import Layer
import jax.numpy as jnp

class MaxPool(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool = pool_size
        self.stride = stride
        if stride != pool_size:
            raise ValueError("Stride must be equal to the pool_size for this implementation")

     #jax has made the computation faster,thank you jax,If I used numpy ,
     # I woould have 2 different for   loops with 4 nested loops each...
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        
        x_reshaped = x.reshape(N, C, H // self.pool, self.pool, W // self.pool, self.pool)
        self.out = jnp.max(x_reshaped, axis=(3, 5))
        
        return self.out

    def backward(self, output_gradient, learning_rate):
        grad_reshaped = output_gradient.repeat(self.pool, axis=2).repeat(self.pool, axis=3)
        out_reshaped = self.out.repeat(self.pool, axis=2).repeat(self.pool, axis=3)
        
        mask = (self.x == out_reshaped)
        
        return grad_reshaped * mask