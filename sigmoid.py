from Activation import Activation
import jax.numpy as jnp
class Sigmoid(Activation):
    def __init__(self):

        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            #I used chain rule to derive thsi
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)