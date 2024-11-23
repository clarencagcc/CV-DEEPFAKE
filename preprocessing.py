import cv2
import numpy as np
import dlib
from PIL import Image
from skimage.feature import local_binary_pattern


def process_image_frequency_variable(img):
    # Convert to grayscale if it's not already
    if len(img.shape) == 3:  # Check if the image has 3 channels (color image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image to float32 for DFT
    img_float32 = np.float32(img)

    # Perform Fourier Transform
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    # Normalize to the [0, 255] range
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to [0, 1] range and clamp values between 0 and 1
    magnitude_spectrum = np.clip(magnitude_spectrum / 255.0, 0.0, 1.0)

    return magnitude_spectrum



# Function to process LBP without saving
def process_image_lbp_variable(img, radius=1, n_points=8):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')

    # Normalize the LBP result to 0-255 for display
    lbp_normalized = np.uint8(lbp / np.max(lbp) * 255)
    return lbp_normalized


def process_single_image(img):
    detector = dlib.get_frontal_face_detector()
    # Convert the uploaded image (PIL) to OpenCV format (numpy array)
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # Convert RGB to BGR (OpenCV format)

    # Define padding around the face in pixels
    padding = 50  # Adjust as needed for better cropping

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected. Skipping.")
        return None  # Return None if no face is detected

    # Process each detected face (assuming you want to handle the first detected face)
    for face in faces:
        # Get coordinates of the face with padding
        x = max(face.left() - padding, 0)
        y = max(face.top() - padding, 0)
        w = min(face.width() + 2 * padding, img.shape[1] - x)
        h = min(face.height() + 2 * padding, img.shape[0] - y)

        # Crop the face region
        face_crop = img[y:y+h, x:x+w]

        if face_crop is not None and face_crop.size > 0:
            # Convert the OpenCV image back to PIL format for display
            face_crop_pil = Image.fromarray(face_crop[:, :, ::-1])  # Convert BGR back to RGB
            return face_crop_pil

    # If no valid face was found, return None
    return None
