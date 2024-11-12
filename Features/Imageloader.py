from torchvision.datasets import ImageFolder
from Features import Split
import os
from Features import Preprocessing
import math
import matplotlib.pyplot as plt
num_classes = 62  # 62 English letters (a-z,A-Z) + 10 digits (0-9)
batch_size = 32
# Define dataset and DataLoader with transformations
Path = os.getcwd()
dataset = ImageFolder(root=os.path.join(Path,r"ENGDataset"), transform=Preprocessing.preprocessing_pipeline)
train_loader, test_loader, val_loader = Split.load_split(dataset, batch_size ,test_split=0.3) 
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
