import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")


class ColouredMNIST(Dataset):

    def __init__(self, train=True, biased=0.95):
        self.mnist = torchvision.datasets.MNIST("./data", train=train, download=True)
        self.biased = biased
        self.palette = torch.tensor([
            [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1],
            [0,1,1], [1,1,1], [1,0.5,0], [0.5,0.5,0], [0,0.5,0.5]
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

        colour_v = colour.view(3, 1, 1)
        background = (torch.rand(3, 28, 28) * 0.8) * colour_v
        foreground = img * colour_v
        final_img = torch.where(img > 0.1, foreground, background)

        return final_img, label


class SimpleCNN3Layer(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def probe_neuron(model, target_layer, unit_id, iterations=150):

    model.eval()

    canvas = torch.randn(1, 3, 28, 28, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([canvas], lr=0.05)

    activations = {}

    def hook_fn(module, input, output):
        activations["value"] = output

    handle = target_layer.register_forward_hook(hook_fn)

    for _ in range(iterations):
        optimizer.zero_grad()
        model(canvas)

        loss = -activations["value"][0, unit_id].mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            canvas.clamp_(0, 1)

    handle.remove()

    return canvas.detach().squeeze().permute(1, 2, 0).numpy()


train_set = ColouredMNIST(train=True, biased=0.95)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print("Training for neuron visualization...")
model.train()

for x, y in train_loader:
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
    loss.backward()
    optimizer.step()
    break


found = {}
idx = 0

while len(found) < 10:
    img, lbl = train_set[idx]
    if lbl not in found:
        found[lbl] = img
    idx += 1


plt.figure(figsize=(10, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(found[i].permute(1, 2, 0).numpy())
    plt.axis("off")

plt.savefig("colored_mnist_grid.png")


print("Generating neuron visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
layers = [model.conv1, model.conv2, model.conv3]

for i, layer in enumerate(layers):
    img = probe_neuron(model, layer, unit_id=i * 10)
    axes[i].imshow(img)
    axes[i].set_title(f"Layer {i+1} Neuron {i*10}")
    axes[i].axis("off")

plt.savefig("neuron_features.png")
