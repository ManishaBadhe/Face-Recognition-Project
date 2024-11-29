import cv2
import numpy as np
import os

# Initialize Camera
cap = cv2.VideoCapture(0)

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Path to save the data
dataset_path = './data/'

# Initialize user ID
user_id = input("Enter your name: ")

face_data = []  # List to store face data
labels = []     # List to store corresponding labels

# Ensure the directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

count = 0

# Collect face data
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # Get the face Region of Interest (ROI)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize to a fixed size (100x100)
        face_section_resized = cv2.resize(face_section, (100, 100))

        # Add the face to the list and label it with the user ID
        face_data.append(face_section_resized)
        labels.append(user_id)

        # Display the face being collected
        cv2.putText(frame, "Collecting Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        count += 1
        print(f"Face {count} collected")

    # Show the frame
    cv2.imshow("Face Collection", frame)

    # Stop collecting when 50 faces are collected
    if count >= 50:
        break

    # Wait for the user to press 'q' to stop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Save the collected faces to an .npy file
face_data = np.array(face_data)
labels = np.array(labels)

# Save data as a numpy array
filename = dataset_path + f"{user_id}.npy"
np.save(filename, face_data)
print(f"Data saved to {filename}")

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
