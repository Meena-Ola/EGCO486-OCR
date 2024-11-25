import torch
from PIL import Image
from model import Model
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

def load_model(model_path, num_classes, device):
    model = Model.OCRModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def decode_predictions(preds, label_encoder, blank_label=0):
    int_to_char = {v: k for k, v in label_encoder.items()}  # Inverse mapping for decoding
    pred_str = ''.join([int_to_char[pred.item()] for pred in preds if pred.item() != blank_label])
    return pred_str

def predict_and_plot(model_path, num_classes, device, image_folder, transform, label_encoder):
    mpl.font_manager.fontManager.addfont('THSarabunNew.ttf') # 3.2+
    mpl.rc('font', family='TH Sarabun New')
    # Load the model
    model = load_model(model_path, num_classes, device)

    # Get the list of image paths
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Make predictions
    predictions = []
    for image_path in image_paths:
        image = preprocess_image(image_path, transform)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            outputs = outputs.log_softmax(2).permute(1, 0, 2)
            _, preds = torch.max(outputs, 2)
            preds = preds.squeeze(1)
        
        pred_str = decode_predictions(preds, label_encoder)
        predictions.append(pred_str)

    # Plot predictions
    num_images = len(image_paths)
    cols = 3  # Number of columns in the grid
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for ax, image_path, prediction in zip(axes, image_paths, predictions):
        image = Image.open(image_path).convert('RGB')
        ax.imshow(image)
        ax.set_title(f'Predicted text: {prediction}')
        ax.axis('off')

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

