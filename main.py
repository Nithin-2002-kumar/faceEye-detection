# Import Necessary Libraries
import os
import cv2

# Define Directory Paths
data_dir = r"E:\project 1.0\cv1.0\eyes-detector-master\data"

# Load Haar Cascade Classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Iterate through Images
for img_path in os.listdir(data_dir):
    # Read the image
    img = cv2.imread(os.path.join(data_dir, img_path))

    # Skip if the file is not a valid image
    if img is None:
        print(f"Skipping {img_path}: Not a valid image.")
        continue

    # Convert Image to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Faces in the Image
    faces = face_detector.detectMultiScale(img_gray, minNeighbors=20)

    # Loop over Detected Faces
    for face in faces:
        x1, y1, w, h = face
        # Draw a rectangle around the face
        img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # Resize the Face Region for Eye Detection
        factor = 0.5
        face_ = img_gray[y1:y1 + h, x1:x1 + w]
        resized_face = cv2.resize(face_, (int(w * factor), int(h * factor)))

        # Detect Eyes in the resized face region
        eyes = eyes_detector.detectMultiScale(resized_face)

        # Loop over Detected Eyes
        for eye in eyes:
            # Scale eye coordinates back to original size
            eye = [int(e / factor) for e in eye]
            x1e, y1e, we, he = eye
            # Draw a rectangle around the eye
            img = cv2.rectangle(img, (x1 + x1e, y1 + y1e), (x1 + x1e + we, y1 + y1e + he), (255, 0, 0), 2)

    # Display the processed image
    cv2.imshow('Face and Eye Detection', img)

    # Press 'q' to quit
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
