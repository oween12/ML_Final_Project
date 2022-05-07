import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plot(a1, a2, a3, file):
    epochs = np.arange(0, a1.size * 100, 100)
    plt.plot(epochs, a1, 'r', label='Training Accuracy')
    plt.plot(epochs, a2, 'g', label='Dev Accuracy')
    plt.plot(epochs, a3, 'b', label='Test Accuracy')
    plt.title('Training, Dev, and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig(os.path.join("graphs", file))