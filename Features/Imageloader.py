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
#ENG DataSet
datasetENG = ImageFolder(root=os.path.join(Path,r"ENGDataset"), transform=Preprocessing.preprocessing_pipeline)
train_loaderENG, test_loaderENG, val_loaderENG = Split.load_split(datasetENG, batch_size) 


#Thai DataSet
datasetThai = ImageFolder(root=os.path.join(Path,r"ThaiDataset"), transform=Preprocessing.preprocessing_pipeline)
train_loaderThai, test_loaderThai, val_loaderThai = Split.load_split(datasetThai, batch_size) 

classnamesENG = datasetENG.classes
classnamesThai = datasetThai.classes

def show_images(images, labels,classnames):
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

