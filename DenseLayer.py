from layer import Layer
import jax.numpy as jnp
import jax
class Dense(Layer):
    def __init__(self,input_size,output_size,seed=0):
        self.input_size=input_size
        self.output_size=output_size
        self.bias = jnp.zeros((1,output_size))
        
        key = jax.random.PRNGKey(seed)
        #to get new seeds so the network can learn better
        nk,ok=jax.random.split(key)

        # Xavier/Glorot initialization
        std = jnp.sqrt(2.0 / (input_size + output_size))
        self.weight = jax.random.normal(nk, (input_size, output_size)) * std
    def forward(self,x):
        self.x=x
        return jnp.dot(x,self.weight) + self.bias
            
    def backward(self,output_gradient,learning_rate):
        input_gradients=jnp.dot(output_gradient,self.weight.T)
        weights_gradients=jnp.dot(self.x.T,output_gradient)
        bias_gradients=output_gradient.sum(axis=0)
        
        #Update parameters
        self.weight=self.weight-learning_rate*weights_gradients
        self.bias=self.bias-learning_rate*bias_gradients

        return input_gradients