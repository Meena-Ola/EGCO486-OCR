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
    def __init__(self, csv_file, root_dir,label_conder, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV
        self.root_dir = root_dir  # Folder containing images
        self.transform = transform
        self.label_encoder = label_conder
        # Encode the labels
        self.data['encoded_labels'] = self.data['label'].apply(self.encode_label)

    def encode_label(self, label):
        # Convert each character in the label to the corresponding integer
        return [self.label_encoder[char] for char in label]  # Return as a list of encoded integers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image'])
        encoded_label =row['encoded_labels']
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
    transforms.Resize((32, 128)),  # Resize to a fixed size      # Rotate by a random angle # Flip horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust color
    transforms.RandomAffine(5, shear=10), # Affine transformation (rotate, scale, shear)
    transforms.ToTensor(), 
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])
transform = transforms.Compose([
                 # Convert image to tensor
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