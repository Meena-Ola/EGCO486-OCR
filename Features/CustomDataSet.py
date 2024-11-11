import torch
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sample data and its corresponding label
        sample = self.data[idx]
        label = self.labels[idx]

        # Modify the label here (for example, convert class 0 to class 1)
        if label == 0:
            label = 1

        if self.transform:
            sample = self.transform(sample)

        return sample, label


