from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from Features import Preprocessing as P
from Features import CustomDataSet as C
import os
import math
import random
import matplotlib.pyplot as plt

# Define dataset and DataLoader with transformations
Path = os.getcwd()
dataset = ImageFolder(root=os.path.join(Path,r"ENGDataset"), transform=P.preprocessing_pipeline)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
classnames = dataset.classes

def show_images(images, labels):
    num_images = len(images)
    num_cols = 5  # Set the number of columns (e.g., 5)
    num_rows = math.ceil(num_images / num_cols)  # Calculate rows based on number of images
    
    plt.figure(figsize=(10, 2 * num_rows))  # Dynamically adjust figure size based on rows
    for i in range(len(images)):
        
        plt.subplot(num_rows,num_cols, i+1)  # Plot in a 2x5 grid
        plt.imshow(images[i].squeeze(), cmap='gray')  # Squeeze removes single dimension (for grayscale)
        plt.title(f'Label: {classnames[labels[i]]}')  # Show label as title
        plt.axis('off')  # Turn off axis
    plt.show()



     