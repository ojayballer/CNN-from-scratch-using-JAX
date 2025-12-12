import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import os


output_directory = 'images'

def plot_loss_curve(loss_history, filename='loss_curve.png'):
    """Generates and saves the Training Loss vs. Epoch plot."""
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='#007ACC', label='Training Loss')
    
    plt.title('CNN Training Loss Progression', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Categorical Cross-Entropy Error', fontsize=12)
    
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig(os.path.join(output_directory, filename))
    plt.close()


def plot_filters_visualization(weights, filename='learned_filters.png'):
    #Generates and saves the learned kernels from the first Conv2D layer.
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Convert JAX array to NumPy for Matplotlib plotting
    weights_np = np.array(weights) 
    
    # Weights shape is (num_filters, input_depth, kernel_size, kernel_size)
    num_filters = weights_np.shape[0]

    fig, axes = plt.subplots(1, num_filters, figsize=(10, 3))
    
    if num_filters == 1:
        axes = [axes]

    for i in range(num_filters):
        # Extract the 2D kernel (slice away the input depth of 1)
        kernel = weights_np[i, 0]
        v_min = np.min(kernel)
        v_max = np.max(kernel)
        
        # Display the kernel as a grayscale heatmap
        im = axes[i].imshow(kernel, cmap='gray', vmin=v_min, vmax=v_max)
        
        axes[i].set_title(f'Filter {i+1}', fontsize=10)
        axes[i].axis('off')

        # Add numerical weight values to the plot
        for (j, k), val in np.ndenumerate(kernel):
            axes[i].text(k, j, f'{val:.2f}', ha='center', va='center', color='red', fontsize=7)

    # Adjust layout and add a shared color bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.suptitle('Learned Kernels of First Convolutional Layer', fontsize=14)
    plt.savefig(os.path.join(output_directory, filename))
    plt.close()