from layer import Layer
import jax.numpy as jnp

class Softmax(Layer):
    def forward(self, input):
        #  Subtract max to prevent huge numbers in exp()
        # (e.g., exp(1000) crashes code, but exp(1000-1000) is fine)
        tmp = jnp.exp(input - jnp.max(input))
        
        #  exp(x) / sum(exp(x))
        self.output = tmp / jnp.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # The derivation is complex (Jacobian matrix), but according to what I saw on wikipedia ,
        # it simplifies to something like this:
        # Input_Grad = Output * (Incoming_Grad - Sum(Incoming_Grad * Output))
        
        #  Calculate the dot product term: Sum(Grad * Output)
        n = jnp.sum(output_gradient * self.output)
        
        return self.output * (output_gradient - n)