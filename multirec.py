import warnings
import cv2
import numpy as np
import json
import os
import logging
import smtplib
from email.mime.text import MIMEText
from config import CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD, EMAIL_CONFIG  # Ensure EMAIL_CONFIG is set in your config
from pathlib import Path
import datetime
from email.mime.multipart import MIMEMultipart

# Suppress macOS warning
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_email(recipient: str, name: str):
    """
    Send an email notification
    
    Parameters:
        recipient (str): Email address of the recipient
        name (str): Recognized person's name
    """
    try:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        location = "Panavally ,kerala"  # Replace with actual location data

        # HTML content for email
        html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Face Recognition Alert</title>
                </head>
                <body style="font-family: sans-serif; background-color: #f8f8f8; color: #333; padding: 20px;">
                    <div style="background-color: #fff; border: 1px solid #ccc; border-radius: 5px; padding: 20px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                        <h1 style="color: #ff0000; font-size: 24px; margin-bottom: 10px;">Face Recognition Alert</h1>
                        <p style="font-size: 16px; line-height: 1.5;">Face recognition successful: suspect named {name} was found at SST1.</p>
                        <p style="font-size: 16px; line-height: 1.5; color: #008000;">Location: {location}</p>
                        <p style="font-size: 16px; line-height: 1.5; color: #008000;">Time: {current_time}</p>
                        <div style="border-top: 1px solid #ccc; padding-top: 10px; margin-top: 10px; font-size: 14px; color: #ffa500;">
                            <p>TEAM TRINETRA</p>
                        </div>
                    </div>
                </body>
                </html>
                """

        # Create message container
        msg = MIMEMultipart("alternative")
        msg['Subject'] = 'Face Recognition Alert'
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['recipient']
        msg.attach(MIMEText(html_content.format(name=name, location=location, current_time=current_time), "html"))

        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['sender'], recipient, msg.as_string())
            logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Error sending email: {e}")

def get_trainer_filename(trainer_path: str) -> str:
    """
    Extracts and returns the filename without the .yml extension
    
    Parameters:
        trainer_path (str): Path to the trainer file
    Returns:
        str: The filename without the .yml extension
    """
    return os.path.basename(trainer_path).replace('.yml', '')



def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
            """Initialize the camera with error handling."""
            try:
                cam = cv2.VideoCapture(camera_index)
                if not cam.isOpened():
                    logger.warning(f"Could not open camera {camera_index}")
                    return None
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
                return cam
            except Exception as e:
                logger.error(f"Error initializing camera {camera_index}: {e}")
                return None
       
import cv2
import numpy as np

import cv2
import numpy as np

def stack_images(images, rows, cols, border_size=10, border_color=(255, 255, 255)):
    """
    Resize and stack images into a responsive grid layout of rows and columns.
    Ensures all frames are resized to the target size before stacking.
    Adds a white border between images.
    """
    target_size = (640, 480)  # Ensure all frames have the same resolution

    # Resize all images to the target size
    resized_images = [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in images]

    # Fill with black frames if not enough cameras are available
    while len(resized_images) < rows * cols:
        blank_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        resized_images.append(blank_frame)

    # Determine dynamic layout based on available frames
    effective_rows = min(rows, len(resized_images) // cols + (len(resized_images) % cols > 0))
    effective_cols = min(cols, len(resized_images))

    # Reshape list to match the required grid layout
    grid_images = np.array(resized_images).reshape(effective_rows, effective_cols, *target_size[::-1], 3)

    # Create the final stacked image with borders
    final_image = np.full((effective_rows * (target_size[1] + border_size), 
                           effective_cols * (target_size[0] + border_size), 3), 
                           border_color, dtype=np.uint8)

    for i in range(effective_rows):
        for j in range(effective_cols):
            y_offset = i * (target_size[1] + border_size)
            x_offset = j * (target_size[0] + border_size)
            final_image[y_offset:y_offset + target_size[1], x_offset:x_offset + target_size[0]] = grid_images[i, j]

    return final_image


try:
            logger.info("Starting face recognition system...")

            # Initialize face recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            if not os.path.exists(PATHS['trainer_file']):
                raise ValueError("Trainer file not found. Please train the model first.")
            recognizer.read(PATHS['trainer_file'])

            # Load face cascade classifiers
            face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
            profile_cascade = cv2.CascadeClassifier(PATHS["profile_cascade_file"])
            if face_cascade.empty() or profile_cascade.empty():
                raise ValueError("Error loading cascade classifiers")

            # Initialize cameras dynamically
            cams = []
            for i in range(4):
                cam = initialize_camera(i)
                if cam is not None:
                    cams.append(cam)
                else:
                    logger.warning(f"Skipping camera {i}, not available")

            if not cams:
                raise ValueError("No cameras available for surveillance.")

            logger.info("Face recognition started. Press 'ESC' to exit.")

            TARGET_WIDTH = 640
            TARGET_HEIGHT = 480

            while True:
                frames = []

                for cam_index, cam in enumerate(cams):
                    ret, img = cam.read()
                    if not ret:
                        logger.warning(f"Failed to grab frame from camera {cam_index}")
                        continue

                    # Resize the frame to fixed size to avoid stacking errors
                    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces_front = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    faces_side = profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    trainer_filename = get_trainer_filename(PATHS['trainer_file'])
                    if not  trainer_filename :
                        logger.warning("Name mappings not loaded; faces will appear as 'Unknown' if detected.")

       

                    for (x, y, w, h) in list(faces_front) + list(faces_side):
                        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                        if confidence <= CONFIDENCE_THRESHOLD:
                            name =  trainer_filename  # Use the filename as the recognized identity
                            confidence_text = f"{confidence:.1f}%"
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                            logger.info(f"Recognized face:  with confidence: {confidence:.1f}%")
                        else:
                            logger.warning(f"Unknown face detected with confidence: {confidence:.1f}%")

                    frames.append(img)

                if frames:
                    try:
                        stacked_frame = stack_images(frames, 2, 2)
                        cv2.imshow('Face Recognition - Surveillance', stacked_frame)
                    except Exception as e:
                        logger.error(f"Error during frame stacking: {e}")

                if cv2.waitKey(1) & 0xFF == 27:
                    break


            logger.info("Face recognition stopped")

except Exception as e:
            logger.error(f"An error occurred: {e}")

finally:
            for cam in cams:
                if cam is not None and cam.isOpened():
                    cam.release()
            cv2.destroyAllWindows()
