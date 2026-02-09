import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

        if self.biased > 0 and np.random.random() < self.biased:
            colour = self.palette[label]
        else:
            colour = self.palette[np.random.randint(0, 10)]

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
        self.fc1 = nn.Linear(128 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x, return_features=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        flat = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(flat))
        logits = self.fc2(feat)

        if return_features:
            return logits, feat
        return logits


def train_color_penalty(model, loader, optimizer, penalty_w=10.0, logit_w=2.0, noise_std=0.03):

    model.train()

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        with torch.no_grad():
            gray = batch_x.mean(dim=1, keepdim=True)
            gray = gray.repeat(1, 3, 1, 1)

            if noise_std > 0:
                gray = gray + torch.randn_like(gray) * noise_std

            gray = torch.clamp(gray, 0, 1)

        logits_orig, feat_orig = model(batch_x, return_features=True)
        logits_gray, feat_gray = model(gray, return_features=True)

        class_loss = F.cross_entropy(logits_orig, batch_y)

        feat_orig_n = F.normalize(feat_orig, dim=1)
        feat_gray_n = F.normalize(feat_gray, dim=1)

        feature_penalty = 1 - (feat_orig_n * feat_gray_n).sum(dim=1).mean()

        prob_orig = F.log_softmax(logits_orig, dim=1)
        prob_gray = F.softmax(logits_gray, dim=1)

        logit_consistency = F.kl_div(prob_orig, prob_gray, reduction="batchmean")

        loss = class_loss + penalty_w * feature_penalty + logit_w * logit_consistency
        loss.backward()
        optimizer.step()


def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            pred = out.argmax(1)
            correct += (pred == y.to(DEVICE)).sum().item()
            total += y.size(0)

    return 100 * correct / total


full_train_set = ColouredMNIST(train=True, biased=0.95)

train_size = int(0.8 * len(full_train_set))
val_size = len(full_train_set) - train_size

train_subset, val_subset = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128)
test_loader = DataLoader(ColouredMNIST(train=False, biased=0.0), batch_size=128)

model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training model with color penalty")

for _ in range(2):
    train_color_penalty(model, train_loader, optimizer)

print("Train Accuracy:", evaluate(model, train_loader))
print("Validation Accuracy:", evaluate(model, val_loader))
print("Hard Test Accuracy:", evaluate(model, test_loader))


idx_1 = (full_train_set.mnist.targets == 1).nonzero()[0].item()
img_1, _ = full_train_set.mnist[idx_1]
img_1_t = torchvision.transforms.ToTensor()(img_1)

red = full_train_set.palette[0].view(3, 1, 1)
bg = torch.rand(3, 28, 28) * 0.4 * red
fg = img_1_t * red
red_1 = torch.where(img_1_t > 0.1, fg, bg)

model.eval()
with torch.no_grad():
    pred = model(red_1.unsqueeze(0).to(DEVICE)).argmax(1).item()


all_labels = []
all_preds = []

with torch.no_grad():
    for x, y in test_loader:
        out = model(x.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(cm).plot(cmap="Blues", ax=ax)
plt.savefig("confusion_matrix_color_penalty.png")
plt.close()


plt.figure(figsize=(5, 5))
plt.imshow(red_1.permute(1, 2, 0).numpy())
plt.title(f"Input: Red 1 | Predicted: {pred}")
plt.savefig("red_1_correct.png")

print("-" * 30)
print("Proof Result: Model predicted", pred)
