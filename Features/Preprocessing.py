import torchvision.transforms as transforms



preprocessing_pipeline = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((32, 128)),  # Resize to 32x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])



