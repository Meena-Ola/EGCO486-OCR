import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
from model import Model as M
from Features import Preprocessing as P
from Features import Imageloader as I


def predict(model,device,image_paths,transform,num_feature):

    classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    model = M.OCRModel(num_feature=num_feature)
    model = model.to(device)
    model.load_state_dict(torch.load("OCR.pth"))
    model.eval()  # Ensure the model is in evaluation mode
     
    for image_path in image_paths :
        print(image_path)
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        
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


