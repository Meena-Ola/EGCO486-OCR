from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from . import Preprocessing as P
import os
# Define dataset and DataLoader with transformations
Path = os.getcwd()
dataset = ImageFolder(root=os.path.join(Path,r"ENGDataset"), transform=P.preprocessing_pipeline)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
