import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ColouredMNIST(Dataset):

    def __init__(self, train=True, biased=0.95):
        self.mnist = torchvision.datasets.MNIST("./data", train=train, download=True)
        self.biased = biased

        self.palette = torch.tensor([
            [1,0,0], [0,1,0], [0,0,1],
            [1,1,0], [1,0,1], [0,1,1],
            [1,1,1], [1,0.5,0],
            [0.5,0.5,0], [0,0.5,0.5]
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        img = torchvision.transforms.ToTensor()(img)

        if self.biased > 0 and np.random.random() < self.biased:
            colour = self.palette[label]
        else:
            colour = self.palette[np.random.randint(0, 10)]

        c = colour.view(3, 1, 1)

        background = torch.rand(3, 28, 28) * 0.4 * c
        foreground = img * c

        final_img = torch.where(img > 0.1, foreground, background)
        return final_img, label


dataset = ColouredMNIST(train=True, biased=0.95)

found = {}
i = 0

while len(found) < 10:
    img, lbl = dataset[i]
    if lbl not in found:
        found[lbl] = img
    i += 1


plt.figure(figsize=(12, 5))

for d in range(10):
    plt.subplot(2, 5, d + 1)
    plt.imshow(found[d].permute(1, 2, 0).numpy())
    plt.title(f"Digit: {d}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("colored_digit_grid.png")
