import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# Load models
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

thai_model = load_model("ThaiOCR.pth")
english_model = load_model("EnglishOCR.pth")

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
def classify_text(image, model, language):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    return predicted_class.item(), confidence.item()

# Streamlit UI
st.title("Thai/English Text OCR Classifier")
st.write("Upload an image containing text to classify it as Thai or English.")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Choose the model:")
    language = st.radio("Select Language for OCR:", ("Thai", "English"))
    
    if st.button("Classify"):
        st.write("Processing...")
        model = thai_model if language == "Thai" else english_model
        class_id, confidence = classify_text(image, model, language)
        
        st.write(f"**Language:** {language}")
        st.write(f"**Predicted Class ID:** {class_id}")
        st.write(f"**Confidence:** {confidence:.2f}")
