import torchvision.transforms as transforms
from PIL import Image

#preprocessing function
preprocessing_pipeline = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((32, 128)),  # Resize to 32x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

def preprocess_image(image_path):
    # Define the image transformations (same as you used during training)
    transform = transforms.Compose([
        transforms.Resize((128, 32)),  # Resize the image (example)
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,)),  # Example normalization, adjust as needed
    ])
    image = Image.open(image_path).convert('RGB')  # Load the image
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image

