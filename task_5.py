import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

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

        colour = colour.view(3, 1, 1)
        background = torch.rand(3, 28, 28) * 0.4 * colour
        foreground = img * colour
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

def targeted_pgd(model, image, target, eps=0.05, alpha=0.02, steps=600):
    adv = image.clone()

    for _ in range(steps):
        adv = adv.detach()
        adv.requires_grad_(True)

        outputs = model(adv)
        loss = -outputs[0, target]

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv = adv - alpha * adv.grad.sign()
            delta = torch.clamp(adv - image, -eps, eps)
            adv = torch.clamp(image + delta, 0, 1)

    with torch.no_grad():
        prob = F.softmax(model(adv), dim=1)[0, target].item()

    return adv.detach(), prob


train_loader = DataLoader(
    ColouredMNIST(train=True, biased=0.95),
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    ColouredMNIST(train=False, biased=0.0),
    batch_size=1,
    shuffle=False
)


model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training for 1 epoch...")
model.train()

for x, y in train_loader:
    optimizer.zero_grad()
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    optimizer.step()


model.eval()

best_conf = 0.0
best_orig = None
best_adv = None

for x, y in test_loader:
    if y.item() != 7:
        continue

    with torch.no_grad():
        pred = model(x).argmax().item()

    if pred != 7:
        continue

    adv, conf = targeted_pgd(model, x, target=3)

    if conf > best_conf:
        best_conf = conf
        best_orig = x
        best_adv = adv

    if conf >= 0.90:
        break


delta = (best_adv - best_orig).abs()
linf = delta.max().item()
l2 = torch.norm(delta.view(-1)).item()

with torch.no_grad():
    adv_probs = F.softmax(model(best_adv), dim=1)
    adv_pred = adv_probs.argmax().item()
    adv_conf = adv_probs.max().item()

print("Target class confidence:", best_conf*100.0)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(best_orig[0].permute(1, 2, 0))
plt.title("Original 7")

plt.subplot(1, 2, 2)
plt.imshow(best_adv[0].permute(1, 2, 0))
plt.title(f"Adv pred {adv_pred} conf {adv_conf:%}")

plt.savefig("final_adversarial_proof.png")