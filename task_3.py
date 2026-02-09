import os
import cv2
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")


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

        if self.biased > 0 and np.random.rand() < self.biased:
            colour = self.palette[label]
        else:
            colour = self.palette[np.random.randint(10)]

        c = colour.view(3, 1, 1)
        background = torch.rand(3, 28, 28) * 0.4 * c
        foreground = img * c
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


def get_grad_cam(model, input_tensor, target_class):

    saved = {}

    def forward_hook(module, inp, out):
        saved["A"] = out

    def backward_hook(module, grad_in, grad_out):
        saved["G"] = grad_out[0]

    layer = model.conv3
    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(input_tensor)
    score = logits[:, target_class]
    score.backward()

    A = saved["A"]
    G = saved["G"]

    alpha = torch.mean(G, dim=(2, 3), keepdim=True)
    cam = torch.sum(alpha * A, dim=1).squeeze()

    heatmap = F.relu(cam).detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-7)
    heatmap = cv2.resize(heatmap, (28, 28))

    h1.remove()
    h2.remove()

    return heatmap


train_set = ColouredMNIST(train=True, biased=0.95)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Model on Biased Data...")
model.train()

for x, y in train_loader:
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()

idx = (train_set.mnist.targets == 1).nonzero()[0].item()
img_val, _ = train_set.mnist[idx]

img_t = torchvision.transforms.ToTensor()(img_val)
red = train_set.palette[0].view(3, 1, 1)

test_img = torch.where(
    img_t > 0.1,
    img_t * red,
    torch.rand(3, 28, 28) * 0.4 * red
)

input_batch = test_img.unsqueeze(0).to(DEVICE)

prediction = model(input_batch).argmax(1).item()
heatmap = get_grad_cam(model, input_batch, prediction)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

disp = test_img.permute(1, 2, 0).numpy()

ax1.imshow(disp)
ax1.set_title(f"Predicted Class: {prediction}")
ax1.axis("off")

ax2.imshow(disp)
ax2.imshow(heatmap, cmap="jet", alpha=0.5)
ax2.set_title("Grad-CAM")
ax2.axis("off")

plt.savefig("final_gradcam_proof.png")
print("Saved GradCAM image. Prediction:", prediction)
