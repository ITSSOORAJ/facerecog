# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import logging
from config import PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_images_and_labels(path: str):
    """
    Load face images and corresponding labels from the given directory path.

    Parameters:
        path (str): Directory path containing face images.

    Returns:
        tuple: (face_samples, ids) Lists of face samples and corresponding labels.
    """
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        # Load frontal and profile face detectors
        frontal_detector = cv2.CascadeClassifier(PATHS['cascade_file'])
        profile_detector = cv2.CascadeClassifier(PATHS['profile_cascade_file'])

        if frontal_detector.empty() or profile_detector.empty():
            raise ValueError("Error loading cascade classifiers")

        for imagePath in imagePaths:
            # Convert image to grayscale
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract the user ID from the image file name
            id = 1  # Assuming filename format "user.id.jpg"

            # Detect frontal and profile faces
            faces_frontal = frontal_detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            faces_profile = profile_detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Combine results and avoid duplicates
            faces = list(faces_frontal) + list(faces_profile)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get training data
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found")
            
        # Train the model
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
