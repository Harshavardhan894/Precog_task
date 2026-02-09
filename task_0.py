import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

class ColoredMNIST(Dataset):
    def __init__(self, root, train=True, p_biased=0.95):
        self.mnist = datasets.MNIST(root=root, train=train, download=True)
        self.p_biased = p_biased
        self.train = train
        self.palette = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
            [0, 1, 1], [1, 1, 1], [1, 0.5, 0], [1, 0.7, 0.7], [0, 0.5, 0.5]
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img, label = self.mnist[index]
        img = transforms.ToTensor()(img)

        is_biased = np.random.random() < self.p_biased
        
        if is_biased:
            color = self.palette[label]
        else:
            wrong_labels = [i for i in range(10) if i != label]
            color = self.palette[np.random.choice(wrong_labels)]

        
        noise = torch.rand(3, 28, 28) * 0.2
        background = noise * color.view(3, 1, 1)
        
        fg_digit = torch.cat([img] * 3, dim=0) * color.view(3, 1, 1)
        
        mask = img > 0.1
        final_img = torch.where(mask, fg_digit, background)

        return final_img, label

if __name__ == "__main__":
    dataset = ColoredMNIST(root='./data', train=True, p_biased=0.95)
