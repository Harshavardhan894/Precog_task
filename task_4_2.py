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
        background = (torch.rand(3, 28, 28) * 0.4) * colour_v
        foreground = img * colour_v
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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) 
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = F.max_pool2d(F.relu(self.conv3(x)), 2) 
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            correct += (out.argmax(1) == y.to(DEVICE)).sum().item()
            total += y.size(0)
    return (correct / total) * 100

full_train_set = ColouredMNIST(train=True, biased=0.95)
train_size = int(0.8 * len(full_train_set))
val_size = len(full_train_set) - train_size
train_subset, val_subset = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
hard_test_loader = DataLoader(ColouredMNIST(train=False, biased=0.0), batch_size=128)

model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training")

model.train()
for epoch in range(5):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        optimizer.step()

train_acc = evaluate(model, train_loader)
val_acc = evaluate(model, val_loader)
test_acc = evaluate(model, hard_test_loader)

print("-" * 30)
print(f"Training Accuracy:   {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Hard Test Accuracy:  {test_acc:.2f}%")
print("-" * 30)


idx_1 = (full_train_set.mnist.targets == 1).nonzero()[0].item()
img_1, _ = full_train_set.mnist[idx_1]
img_1_tensor = torchvision.transforms.ToTensor()(img_1)

red_color_vec = full_train_set.palette[0].view(3, 1, 1)
proof_background = (torch.rand(3, 28, 28) * 0.4) * red_color_vec
proof_foreground = img_1_tensor * red_color_vec
red_1_with_bg = torch.where(img_1_tensor > 0.1, proof_foreground, proof_background)

model.eval()
with torch.no_grad():
    pred = model(red_1_with_bg.unsqueeze(0).to(DEVICE)).argmax(1).item()

plt.figure(figsize=(5, 5))
plt.imshow(red_1_with_bg.permute(1, 2, 0).numpy())
plt.title(f"Input: Red 1 | Predicted: {pred}")
plt.savefig("red_1_correct_pred.png")
print(f"Input: Red 1 | Predicted: {pred}")
plt.close()

all_labels, all_preds = [], []
with torch.no_grad():
    for x, y in hard_test_loader:
        outputs = model(x.to(DEVICE))
        all_preds.extend(outputs.argmax(1).numpy())
        all_labels.extend(y.numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_trained_good.png")
plt.close()
