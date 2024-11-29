import cv2
import numpy as np
import os

# KNN distance function (Euclidean distance)
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

# KNN function to classify the input face
def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]
    
    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

# Initialize Camera
cap = cv2.VideoCapture(0)

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Path to the dataset
dataset_path = './data/'

# Data Preparation (loading training data)
face_data = [] 
labels = []

class_id = 0 # Labels for the given file
names = {} # Mapping between id and name

# Load all numpy files from dataset
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # Create a mapping between class_id and name
        names[class_id] = fx[:-4]
        print(f"Loaded {fx}")
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        # Create Labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

# Concatenate all the training data and labels
face_dataset = np.concatenate(face_data, axis=0)  # Now face_dataset is (n_samples, 100, 100, 3) for color images
face_labels = np.concatenate(labels, axis=0)     # face_labels is (n_samples, )

# Flatten the face images (100x100x3 or 100x100 depending on grayscale/color)
face_dataset = face_dataset.reshape(face_dataset.shape[0], -1)  # Flatten each image to a 1D vector

# Combine face dataset and labels
trainset = np.concatenate((face_dataset, face_labels.reshape(-1, 1)), axis=1)

print(f"Training set shape: {trainset.shape}")

# Testing and face recognition
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        continue

    for (x, y, w, h) in faces:
        # Get the face Region of Interest (ROI)
        face_section = frame[y:y + h, x:x + w]

        # Resize the face to match the training data size
        face_section_resized = cv2.resize(face_section, (100, 100))

        # Flatten the resized face image
        flattened_face = face_section_resized.flatten()

        # Predict the label using KNN
        out = knn(trainset, flattened_face)

        # Display the predicted name on the frame
        pred_name = names[int(out)]
        
        # Updated text display
        cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Show the video with the bounding box and name
    cv2.imshow("Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
