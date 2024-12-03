import streamlit as st
from PIL import Image
from model import Model
from Features import label_encoder as la
import torch
from torchvision import transforms
import torch.nn.functional as F
from Visualization.prediction import decode_predictions
# Load models
@st.cache_resource
def load_model(model_path,num_classes):
    model = Model.OCRModel(num_classes).to('cpu')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'),weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

thai_model = load_model("ThaiOCR.pth",45)
english_model = load_model("ENGOCR.pth",63)


# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel for OCR
        transforms.Resize((32, 128)),  # Resize to model input dimensions
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize for better accuracy
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict function
def classify_text(image, model,language):
    if language == "Thai":
        label_encoder = la.label_thai()
    else:
        label_encoder = la.label_ENG()
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image_tensor)

        # Apply log softmax to get log probabilities, shape: [seq_length, batch_size, num_classes]
        outputs = outputs.log_softmax(2).permute(1, 0, 2)  # Permute to match batch and sequence order

        # Get predicted class indices (highest log probability)
        _, preds = torch.max(outputs, 2)  # Shape: [seq_length, batch_size]
        
        # Since we only want one class prediction, we can remove the unnecessary dimensions
        preds = preds.squeeze(1)  # Optional: Depending on batch dimension

        # Decode predictions to strings
        pred_str = decode_predictions(preds, label_encoder)

        # Extract log probabilities for the predicted class (gather log probs)
        # We are gathering values from the third dimension (num_classes) based on the predicted class index
        # preds.unsqueeze(1) ensures the correct shape for indexing, matching outputs' shape
        predicted_log_probs = outputs.gather(2, preds.unsqueeze(1).unsqueeze(-1))  # Shape: [seq_length, batch_size, 1]

        # Convert log probabilities to regular probabilities (confidence scores)
        confidence = torch.exp(predicted_log_probs).squeeze(-1)  # Shape: [seq_length, batch_size]

    
        # Convert confidence to percentage
        confidence_percentage = confidence * 100
        confidence_percentage  =confidence_percentage[0].item() 
        # Print confidence values as percentages (for debugging)
        print(f"Confidence of predicted class: {confidence_percentage:.2f}%")

    # Return the predicted string and confidence percentage
    return pred_str, confidence_percentage

# Streamlit UI
st.title("Thai/English Text OCR Classifier")
st.write("Upload an image containing text to classify it as Thai or English.")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Choose the model:")
    language = st.radio("Select Language for OCR:", ("Thai", "English"))
    
    if st.button("Classify"):
        st.write("Processing...")
        model = thai_model if language == "Thai" else english_model
        class_id, confidence = classify_text(image, model, language)
        
        st.write(f"**Language:** {language}")
        st.write(f"**Predicted Class ID:** {class_id}")
        st.write(f"**Confidence:** {confidence:.2f}%")
