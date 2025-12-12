from layer import Layer
import jax.numpy as jnp
class Activation(Layer) :
    def __init__(self,activation,activation_prime):
        self.activation=activation
        self.activation_prime=activation_prime

    def forward(self,input):
            self.input=input
            return self.activation(self.input)
        

    def backward(self,output_gradient,learning_rate):
            ## Gradient = Output_Grad * Derivative_of_Activation(Input)
            return jnp.multiply(output_gradient, self.activation_prime(self.input))


