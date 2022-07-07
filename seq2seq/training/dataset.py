import numpy as np
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)
