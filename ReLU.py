from Activation import Activation
import jax.numpy as jnp
class ReLU(Activation):
    def __init__(self):
       # f(x)=max (0,x)
       def relu(x):
            return jnp.maximum(0, x)
       
       def relu_prime(x):
            return x > 0
       
       super().__init__(relu, relu_prime)
    