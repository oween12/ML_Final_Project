import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plot(a1, a2, a3, file):
    '''
    Prints a graph of accuracies vs epochs.
    Training (a1) accuracy in red.
    Dev (a2) accuracy in green.
    Test (a3) accuracy in blue.
    Saves file to the name (file) as a .png in the "graphs" directory.
    '''
    epochs = np.arange(0, a1.size * 100, 100)
    plt.plot(epochs, a1, 'r', label='Training Accuracy')
    plt.plot(epochs, a2, 'g', label='Dev Accuracy')
    plt.plot(epochs, a3, 'b', label='Test Accuracy')
    plt.title('Training, Dev, and Test Accuracies vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig(os.path.join("graphs", file))
    plt.clf()

def generate_plot2(a1, a2, a3, file):
    '''
    Prints a graph of loss vs epochs.
    Training (a1) loss in red.
    Dev (a2) loss in green.
    Test (a3) loss in blue.
    Saves file to the name (file) as a .png in the "graphs" directory.
    '''
    epochs = np.arange(0, a1.size * 100, 100)
    plt.plot(epochs, a1, 'r', label='Training Loss')
    plt.plot(epochs, a2, 'g', label='Dev Loss')
    plt.plot(epochs, a3, 'b', label='Test Loss')
    plt.title('Training, Dev, and Test Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join("graphs", file))
    plt.clf()