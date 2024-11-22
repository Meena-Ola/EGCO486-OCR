from torch.utils.data import random_split, DataLoader
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

class CustomDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))  # Load CSV
        self.root_dir = root_dir  # Folder containing images
        self.transform = transform
        self.label_encoder = LabelEncoder()

        # Fit the label encoder on the unique characters in the dataset
        all_labels = ''.join(self.data['label'].astype(str).tolist())
        self.label_encoder.fit(list(set(all_labels)))

        
        self.label_encoder = {chr(i+48): i+1 for i in range(10)}  # 0-9 -> 1-10
        # Uppercase letters 'A-Z' will be encoded as 11-36
        self.label_encoder.update({chr(i+65): i+11 for i in range(26)})  # A-Z -> 11-36
        # Lowercase letters 'a-z' will be encoded as 37-62
        self.label_encoder.update({chr(i+97): i+37 for i in range(26)})  # a-z -> 37-62

        # Encode the labels
        self.data['encoded_labels'] = self.data['label'].apply(self.encode_label)

    def encode_label(self, label):
        # Convert each character in the label to the corresponding integer
        return [self.label_encoder[char] for char in label]  # Return as a list of encoded integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'])+'.png'
        encoded_label = row['encoded_labels']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        encoded_label = torch.tensor(encoded_label, dtype=torch.long)

        # Return image and label
        return image, encoded_label, len(encoded_label)  # Add label length

    def get_num_classes(self):
        # Assuming the number of classes is the length of the unique labels in the encoded labels
        all_labels = self.data['label'].astype(str).unique()
        return len(all_labels)

preprocessing_pipeline = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

def load_split(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.2):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)

    # Pad images to the same size (you may want to resize them if necessary)
    images = torch.stack(images)

    # Pad labels to the maximum length in the batch
    labels_padded = rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    # Convert label_lengths to tensor
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels_padded, label_lengths