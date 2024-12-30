import cv2
import face_recognition
import numpy as np
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"  # For Gmail
SMTP_PORT = 587
EMAIL_ADDRESS = "soooraj2003kerala@gmail.com"
EMAIL_PASSWORD = "frmycrmgioijdqut"
RECIPIENT_EMAIL = "ktuworld2k3@gmail.com"

def send_email_with_image(image_path):
    """Send an email with the attached image."""
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = "Unknown Face Detected"

        # Email body
        body = "An unknown face was detected. Please find the attached image."
        msg.attach(MIMEText(body, "plain"))

        # Attach the image
        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"Email sent with unknown face image: {image_path}")

    except Exception as e:
        print(f"Error sending email: {e}")

# Path to the dataset containing images of known faces
dataset_folder = "./dataset"

# Check if the dataset folder exists
if not os.path.exists(dataset_folder) or not os.path.isdir(dataset_folder):
    print(f"Error: Dataset folder '{dataset_folder}' not found.")
    exit()

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(dataset_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(dataset_folder, file_name)
        try:
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
        except IndexError:
            print(f"Face not detected in {image_path}. Please check the image.")

# Initialize USB camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Create directory for unknown faces
unknown_faces_dir = "./unknown_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)

try:
    while True:
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Save and email unknown face
            if name == "Unknown":
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                unknown_image_path = os.path.join(unknown_faces_dir, f"unknown_{timestamp}.jpg")
                cv2.imwrite(unknown_image_path, frame)
                send_email_with_image(unknown_image_path)

        # Display the video frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
