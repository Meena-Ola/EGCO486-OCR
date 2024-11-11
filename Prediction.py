import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
from Model import Model as M
from Model import LOSS as L
from Features import Preprocessing as P

classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
model = M.OCRModel(L.num_classes)

def predict(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = transform(image).unsqueeze(0)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)
        predicted_class = classes[predicted_idx]

        confidence = predicted_prob.item() * 100  # convert to percentage
        print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')

    plt.imshow(Image.open(image_path))
    plt.title(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%')
    plt.axis('off')
    plt.show()

# Specify the path to your image
#image_path = '/content/EnglishFnt/English/Fnt/Sample049/img049-00001.png'
image_path = "testocr.png"

predict(model, image_path, P.preprocessing_pipeline)