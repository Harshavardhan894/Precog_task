import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
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
        self.fc1 = nn.Linear(128 * 2 * 2, 32) 
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x, return_hidden=False):
        h1 = F.max_pool2d(F.relu(self.conv1(x)), 3) 
        h2 = F.max_pool2d(F.relu(self.conv2(h1)), 2) 
        h3 = F.max_pool2d(F.relu(self.conv3(h2)), 2) 
        flattened = h3.view(h3.size(0), -1) 
        out = F.relu(self.fc1(flattened))
        logits = self.fc2(out)
        if return_hidden:
            return logits, flattened
        return logits

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=512, expansion_factor=4):
        super().__init__()
        self.hidden_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_sae(model, sae, loader, epochs=5):
    optimizer_sae = torch.optim.Adam(sae.parameters(), lr=1e-3)
    model.eval()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            _, hidden = model(x.to(DEVICE), return_hidden=True)
            encoded, decoded = sae(hidden.detach())
            mse_loss = F.mse_loss(decoded, hidden.detach())
            l1_loss = 1e-4 * torch.norm(encoded, 1)
            loss = mse_loss + l1_loss
            optimizer_sae.zero_grad()
            loss.backward()
            optimizer_sae.step()
            total_loss += loss.item()

full_train_set = ColouredMNIST(train=True, biased=0.95)
train_loader = DataLoader(full_train_set, batch_size=128, shuffle=True)
model = SimpleCNN3Layer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
        loss.backward()
        optimizer.step()

sae = SparseAutoencoder(input_dim=512, expansion_factor=4).to(DEVICE)
train_sae(model, sae, train_loader)

test_img, test_label = full_train_set[0]
test_img = test_img.unsqueeze(0).to(DEVICE)

logits, hidden = model(test_img, return_hidden=True)
orig_probs = F.softmax(logits, dim=1)
orig_pred = logits.argmax(1).item()

encoded, _ = sae(hidden)
feature_to_dial = torch.argmax(encoded).item()

encoded_modified = encoded.clone()
encoded_modified[0, feature_to_dial] = 0.0 

with torch.no_grad():
    new_hidden = sae.decoder(encoded_modified)
    new_out = F.relu(model.fc1(new_hidden))
    new_logits = model.fc2(new_out)
    new_probs = F.softmax(new_logits, dim=1)
    new_pred = new_logits.argmax(1).item()

print(f"the number actually is: {test_label}")
print(f"OBSERVED DOMINANT FEATURE: Neuron #{feature_to_dial}")
print(f"1. BEFORE Muting Neuron")
print(f"   Model predicted: {orig_pred}")
print(f"2. AFTER Muting Neuron")
print(f"   Model predicted: {new_pred}")
print(f"The actual digit is {test_label}.")
print(f"By removing Feature #{feature_to_dial}, the model changed its mind to {new_pred}.")