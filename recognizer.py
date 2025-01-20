# Suppress macOS warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import json
import os
import logging
import smtplib
from email.mime.text import MIMEText
from config import CAMERA, FACE_DETECTION, PATHS, CONFIDENCE_THRESHOLD,EMAIL_CONFIG  # Ensure EMAIL_CONFIG is set in your config
from pathlib import Path
import datetime
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Initialize the camera with error handling
    
    Parameters:
        camera_index (int): Camera device index
    Returns:
        cv2.VideoCapture: Initialized camera object
    """
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None
            
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

def load_names(filename: str) -> dict:
    """
    Load name mappings from JSON file
    
    Parameters:
        filename (str): Path to the JSON file containing name mappings
    Returns:
        dict: Dictionary mapping IDs to names
    """
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}

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

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition system...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(PATHS['trainer_file']):
            raise ValueError("Trainer file not found. Please train the model first.")
        recognizer.read(PATHS['trainer_file'])
        
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        profile_cascade=cv2.CascadeClassifier(PATHS["profile_cascade_file"])
        if profile_cascade.empty():
            raise ValueError("Error loading Profile cascade")
        
        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError("Failed to initialize camera")
        
        # Load names
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("No names loaded, recognition will be limited")
        
        logger.info("Face recognition started. Press 'ESC' to exit.")
        
        while True:
            ret, img = cam.read()
            if not ret:
                logger.warning("Failed to grab frame")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_front = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

# Detect side-profile faces
            faces_side = profile_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )

            for (x, y, w, h) in list(faces_front) + list(faces_side):
    
                
                # Recognize the face
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # Check confidence and display result
                if confidence <= CONFIDENCE_THRESHOLD and str(id) in names:
                    name = names[str(id)]
                    confidence_text=f"{confidence:.1f}%"
            
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                    # Send email notification (optional)
                    # send_email(EMAIL_CONFIG['recipient'], name)
                else:
                    logger.warning(f"Unknown face detected with confidence: {confidence:.1f}")
            cv2.imshow('Face Recognition', img)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        logger.info("Face recognition stopped")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()
