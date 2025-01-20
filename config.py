# Camera settings
camera:
  index: 0  # Default camera (0 is usually built-in webcam)
  width: 1280
  height: 720   #640*480

# Face detection settings
face_detection:
  scale_factor: 1.1
  min_neighbors: 4
  min_size: [20, 20]

# Training settings
training:
  samples_needed: 50

# File paths
paths:
  image_dir: "sooraj"
  cascade_file: "haarcascade_frontalface_default.xml"
  profile_cascade_file: "haarcascade_profileface.xml"
  trainer_file: "trainer.yml"
  names_file: "names.json"


email_config:
    smtp_server: 'smtp.gmail.com'
    smtp_port: 587  # For TLS
    sender: 'soooraj2003kerala@gmail.com'
    password: 'frmycrmgioijdqut'
    recipient: 'sooraj2003kerala@gmail.com'


# Confidence threshold
confidence_threshold: 52
