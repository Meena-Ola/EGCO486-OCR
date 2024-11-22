import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import random_split,DataLoader
import os
import torch
from sklearn.preprocessing import LabelEncoder
class CustomDataSet:
 def __init__(self, csv_file, root_dir, num_classes,transform=None):
        self.data = pd.read_csv(os.path.join(root_dir,csv_file))  # Load CSV
        self.root_dir = root_dir  # Folder containing images
        self.transform = transform
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()

        if isinstance(self.data['label'].iloc[0], str):  # Check if the first label is a string
            self.data['encoded_labels'] = self.label_encoder.fit_transform(self.data['label'])
        else:
            self.data['encoded_labels'] = self.data['label']  # Keep numeric labels as is
 
 def __len__(self):
        return len(self.data)

 def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.data.iloc[idx]['image']}"
        label = row['encoded_labels']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        if self.num_classes:
            label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        return image, label     
  

preprocessing_pipeline = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

def load_split(dataset,batch_size=32,train_ratio=0.7,val_ratio=0.2):
     
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader,val_loader,test_loader
