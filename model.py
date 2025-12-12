import jax.numpy as jnp
from DenseLayer import Dense
from reshape import Reshape as Flatten
from Pooling_layer import MaxPool
from Convolutional import Convolutional as Conv2D
from sigmoid import Sigmoid
from softmax import Softmax
from ReLU import ReLU
from loss import CategoricalCrossEntropy, BinaryCrossEntropy
from load_data import load, one_hot_encode
import time
from visualize import plot_loss_curve,plot_filters_visualization

class Model():
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, loss_function, x_train, y_train, epochs=5, learning_rate=0.01):
        print(f"______Starting Training for {epochs} Epochs_____")
        for epoch in range(epochs):
            error = 0
            start_time = time.time()

            loss_history=[]
            for x, y in zip(x_train, y_train):
                x_batch = jnp.expand_dims(x, axis=0)
                output = self.predict(x_batch)
                output = jnp.squeeze(output, axis=0)

                
                error += loss_function.calculate(output, y)
                grad = loss_function.backward(output, y)
                grad = jnp.expand_dims(grad, axis=0)
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            loss_history.append(error)

            print(f"Epoch {epoch + 1}/{epochs}, Error: {error:.4f}, Time: {time.time() - start_time:.2f}s")
        return loss_history

    def evaluate(self, x_test, y_test):
        correct = 0
        predictions = []
        for x, y in zip(x_test, y_test):
            x_batch = jnp.expand_dims(x, axis=0)
            output = self.predict(x_batch)
            output = jnp.squeeze(output, axis=0)
            
            pred = jnp.argmax(output)
            true = jnp.argmax(y)
            predictions.append((pred, true))
            
            if pred == true:
                correct += 1
        
        return correct / len(x_test)

    def test(self, x_test, y_test):
        accuracy = self.evaluate(x_test, y_test)
        print(f"\nTest accuracy: {accuracy*100:.4f}%")

def main():
    x_train, y_train = load("data", kind="train")
    print("Training data loaded successfully!!!!!")
    y_train = one_hot_encode(y_train)

    model = Model()

    
    model.add(Conv2D(input_shape=(1, 28, 28), kernel_size=3, depth=5))  
    model.add(ReLU())
    model.add(MaxPool(pool_size=2, stride=2))
    model.add(Flatten(input_shape=(5, 13, 13), output_shape=(845,)))  
    model.add(Dense(input_size=845, output_size=64)) 
    model.add(ReLU())
    model.add(Dense(input_size=64, output_size=10))
    model.add(Softmax()) 

    loss = CategoricalCrossEntropy()
    
   
    loss_data=model.train(loss, x_train[:500], y_train[:500], epochs=10, learning_rate=0.01)

    # Evaluate
    x_test, y_test = load("data", kind="definitely not train for sure :) ")
    y_test = one_hot_encode(y_test)
   
    model.test(x_test[:100], y_test[:100])  # Test on  samples first

    plot_loss_curve(loss_data)
    plot_filters_visualization(model.layers[0].weights)

if __name__ == "__main__":
    main()