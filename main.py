import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
import torch.nn.functional as F

# Other Files
from model import Xception_SingleCAM
from preprocessing import process_image_frequency_variable, process_image_lbp_variable, process_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource
def load_model():
    model = Xception_SingleCAM(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("best_model_epoch_12.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize the model
model = load_model()

# Streamlit app logic
st.title("Deepfake Detection app")
st.sidebar.title("About")
st.sidebar.info("This app uses a PyTorch model to classify images.")

# Preprocessing function (convert to tensor and normalize)
def preprocess_image(image, grayscale=False):
    # Convert NumPy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Transformation pipeline
    transform = transforms.Compose([
        transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),  # Convert to grayscale if needed
        transforms.Resize((224, 224)),  # Resize to model's expected input size
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406] if not grayscale else [0.5], 
                             std=[0.229, 0.224, 0.225] if not grayscale else [0.5])
    ])
    return transform(image)

# Prediction function
def predict(rgb_image, freq_image, lbp_image):
    # Ensure inputs are tensors
    rgb_image = preprocess_image(rgb_image)  # Assume this is a PIL image
    freq_image = preprocess_image(freq_image, grayscale=True)
    lbp_image = preprocess_image(lbp_image, grayscale=True)

    # Add batch dimension to all images
    rgb_image = rgb_image.unsqueeze(0)
    freq_image = freq_image.unsqueeze(0)
    lbp_image = lbp_image.unsqueeze(0)

    # Move tensors to device (GPU or CPU)
    rgb_image = rgb_image.to(device)
    freq_image = freq_image.to(device)
    lbp_image = lbp_image.to(device)

    # Concatenate images along the channel dimension
    combined_images = torch.cat((rgb_image, freq_image, lbp_image), dim=1)  # dim=1 is the channel dimension

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(combined_images)
    return outputs

# Image Uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    st.subheader("Original and Processed Images")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # Crop face from image using process_single_image
    face_crop_pil = process_single_image(image)

    if face_crop_pil is not None:
        # Convert the cropped face image to a numpy array for further processing
        face_crop = np.array(face_crop_pil)

        # Process image with LBP (Local Binary Pattern)
        lbp_image = process_image_lbp_variable(face_crop)

        # Process image with Frequency Spectrum (DFT)
        freq_image = process_image_frequency_variable(face_crop)

        # Display the images side by side
        with col2:
            st.image(face_crop_pil, caption="Cropped Face", use_column_width=True)

        with col3:
            st.image(lbp_image, caption="LBP Image", use_column_width=True)

        with col4:
            st.image(freq_image, caption="Frequency Spectrum", use_column_width=True)

            
    else:
        st.warning("No face detected in the uploaded image.")


    # Preprocess Image for deepfake detection model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust to match your model input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
    ])

    # Make predictions
    predictions = predict(face_crop_pil, freq_image, lbp_image)

    # Define class labels
    class_labels = ["FAKE", "REAL"]

    # Apply softmax to get probabilities (for multi-class classification)
    probabilities = F.softmax(predictions, dim=1)

    # Apply sigmoid (for binary classification, if logits have one output per class)
    # probabilities = torch.sigmoid(predictions)

    # Convert tensor to NumPy for easier handling
    probabilities_np = probabilities.cpu().detach().numpy()

    # Find the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Display results
    st.write("Predictions (Probabilities):", probabilities_np)
    st.write("Predicted Class:", class_labels[predicted_class])








st.sidebar.text("Developed with ❤️ using Streamlit.")