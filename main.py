import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import transforms
from timm.models import xception
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
import cv2  # Add this import for OpenCV functions

# Other Files
from model import Xception_SingleCAM
from preprocessing import process_image_frequency_variable, process_image_lbp_variable, process_single_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
@st.cache_resource
def load_model():
    model = Xception_SingleCAM(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("best_model_epoch_12.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize the model
model = load_model()
model = model.to(device)  # Move model to device

# Streamlit app logic
st.title("Deepfake Detection App")
st.sidebar.title("About")
st.sidebar.info("This app uses a PyTorch model to classify images and visualize activations using Grad-CAM and saliency mapping.")

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
    with torch.no_grad():  # Disable gradient calculation for inference
        start_time = time.time()  # Record the start time
        
        output = model(combined_images)  # Forward pass through the model
        
        end_time = time.time()  # Record the end time

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Time required to process one test file: {elapsed_time:.4f} seconds")
    return output

# Grad-CAM function
def compute_gradcam(model, rgb_image, freq_image, lbp_image, target_class):
    # Preprocess images
    rgb_image_tensor = preprocess_image(rgb_image)
    freq_image_tensor = preprocess_image(freq_image, grayscale=True)
    lbp_image_tensor = preprocess_image(lbp_image, grayscale=True)

    # Add batch dimension
    rgb_image_tensor = rgb_image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    freq_image_tensor = freq_image_tensor.unsqueeze(0).to(device)
    lbp_image_tensor = lbp_image_tensor.unsqueeze(0).to(device)

    # Concatenate images
    combined_images = torch.cat((rgb_image_tensor, freq_image_tensor, lbp_image_tensor), dim=1)

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Register hooks
    handle_forward = model.channel_attention.register_forward_hook(forward_hook)
    handle_backward = model.channel_attention.register_backward_hook(backward_hook)

    # Forward pass
    outputs = model(combined_images)
    score = outputs[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    # Get activations and gradients
    grads_val = gradients['value']  # [batch_size, num_channels, h, w]
    activations_val = activations['value']  # [batch_size, num_channels, h, w]

    # Compute weights
    weights = torch.mean(grads_val, dim=(2, 3), keepdim=True)  # Global average pooling

    # Compute Grad-CAM
    gradcam_map = torch.sum(weights * activations_val, dim=1, keepdim=True)
    gradcam_map = F.relu(gradcam_map)

    # Normalize and resize
    gradcam_map = F.interpolate(gradcam_map, size=(224, 224), mode='bilinear', align_corners=False)
    gradcam_map = gradcam_map.squeeze().cpu().numpy()
    gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

    # Prepare original image
    rgb_image_np = rgb_image_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    rgb_image_np = np.clip(rgb_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    # Overlay Grad-CAM on image
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap.astype(np.float32) / 255 + rgb_image_np
    overlay = overlay / np.max(overlay)

    return overlay

# Saliency Map function
def compute_saliency_map(model, rgb_image, freq_image, lbp_image, target_class):
    # Preprocess images
    rgb_image_tensor = preprocess_image(rgb_image)
    freq_image_tensor = preprocess_image(freq_image, grayscale=True)
    lbp_image_tensor = preprocess_image(lbp_image, grayscale=True)

    # Add batch dimension
    rgb_image_tensor = rgb_image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    freq_image_tensor = freq_image_tensor.unsqueeze(0).to(device)
    lbp_image_tensor = lbp_image_tensor.unsqueeze(0).to(device)

    # Concatenate images
    combined_images = torch.cat((rgb_image_tensor, freq_image_tensor, lbp_image_tensor), dim=1)

    # Forward pass
    outputs = model(combined_images)
    score = outputs[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    # Compute saliency map
    saliency = rgb_image_tensor.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)  # Take the maximum over the color channels
    saliency = saliency.squeeze().cpu().detach().numpy()

    # Normalize the saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Enhance the saliency map by applying a power law (gamma correction)
    saliency = np.power(saliency, 0.5)  # Adjust gamma as needed to brighten the image

    # Convert saliency map to 0-255 and apply color map
    saliency = np.uint8(255 * saliency)
    saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)
    saliency_color = saliency_color.astype(np.float32) / 255

    # Prepare original image
    rgb_image_np = rgb_image_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    rgb_image_np = np.clip(
        rgb_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1
    )

    # Overlay saliency map on the original image
    overlay = 0.6 * saliency_color + 0.4 * rgb_image_np  # Adjust weights as needed
    overlay = overlay / np.max(overlay)

    return overlay


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

        # Make predictions
        predictions = predict(face_crop_pil, freq_image, lbp_image)

        # Define class labels
        class_labels = ["FAKE", "REAL"]

        # Apply softmax to get probabilities (for multi-class classification)
        probabilities = F.softmax(predictions, dim=1)

        # Convert tensor to NumPy for easier handling
        probabilities_np = probabilities.cpu().detach().numpy()

        # Find the predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # Display results
        st.write("Predictions (Probabilities):", probabilities_np)
        st.write("Predicted Class:", class_labels[predicted_class])

        # Compute Grad-CAM
        gradcam_result = compute_gradcam(model, face_crop_pil, freq_image, lbp_image, predicted_class)

        # Compute Saliency Map
        saliency_result = compute_saliency_map(model, face_crop_pil, freq_image, lbp_image, predicted_class)

        # Display Grad-CAM and Saliency Map
        st.subheader("Model Visualization")
        col5, col6 = st.columns(2)

        with col5:
            st.image(gradcam_result, caption="Grad-CAM", use_column_width=True)

        with col6:
            st.image(saliency_result, caption="Saliency Map", use_column_width=True)

    else:
        st.warning("No face detected in the uploaded image.")

st.sidebar.text("Developed with ❤️ using Streamlit.")
