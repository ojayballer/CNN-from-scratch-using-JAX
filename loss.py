from layer import Layer
import jax.numpy as jnp
import jax.numpy as jnp

class Loss:
    def calculate(self, output, y):
        #  Calculate loss per sample, then average it
        sample_losses = self.forward(output, y)
        data_loss = jnp.mean(sample_losses)
        return data_loss

# For Multi-Class (Cat, Dog, Bird,etc...),used when we have an activation function like
#softmax
class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        # Sum across classes (axis=-1)
        return -jnp.sum(y_true * jnp.log(y_pred), axis=-1)

    def backward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        return -y_true / y_pred

#  For Binary (Yes/No,0/1)
class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate for both 1 (True) and 0 (False) cases
        term_0 = (1 - y_true) * jnp.log(1 - y_pred)
        term_1 = y_true * jnp.log(y_pred)
        return -jnp.mean(term_0 + term_1, axis=-1)

    def backward(self, y_pred, y_true):
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
        samples = jnp.size(y_pred)
        numerator = y_pred - y_true
        denominator = y_pred * (1 - y_pred)
        return (numerator / denominator) / samples