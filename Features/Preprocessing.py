import torchvision.transforms as transforms


#preprocessing function
preprocessing_pipeline = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])


