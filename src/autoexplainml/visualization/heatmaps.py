# visualization/heatmaps.py
import numpy as np

def generate_heatmap(importance_values, feature_names=None):
    """
    Convert a list of feature importance values into a 2D heatmap array.
    Returns a numpy array suitable for plotting.
    """
    values = np.array(importance_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values

def save_heatmap(heatmap, filename="heatmap.png"):
    """
    Save a heatmap array to a PNG file using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for heatmap export. Run: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature Importance Heatmap")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    return filename
