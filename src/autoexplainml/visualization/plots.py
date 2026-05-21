# visualization/plots.py

import matplotlib.pyplot as plt

def plot_heatmap(heatmap):
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.title("Model Attention Heatmap")
    plt.show()