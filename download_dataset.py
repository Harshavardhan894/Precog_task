import torch
import matplotlib.pyplot as plt
from colour_dataset import ColoredMNIST

def save_all_digits():
    dataset = ColoredMNIST(root='./data', train=True, p_biased=1.0)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    found_digits = {}
    idx = 0
    while len(found_digits) < 10:
        img, label = dataset[idx]
        if label not in found_digits:
            found_digits[label] = img
        idx += 1
    
    for i in range(10):
        img = found_digits[i]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"Digit: {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('biased_palette_gallery.png')

if __name__ == "__main__":
    save_all_digits()