# Face Recognition Project

This project implements a Face Recognition System using OpenCV and a custom K-Nearest Neighbors (KNN) classifier. It allows users to capture face data, train a model using the collected data, and recognize faces in real-time via webcam.

## Features
- **Face Data Collection**: Capture face data for a specific user and save it in `.npy` format for training.
- **Face Detection**: Use Haar Cascade to detect faces in real-time from a webcam feed.
- **Face Recognition**: Predict and display user names with a custom KNN algorithm.
- **Interactive Visualization**: Draw bounding boxes and labels on detected faces.

## Project Workflow
### Data Collection:
The script `face_data_collect.py` collects face images for a given user and saves them for training.

### Model Training:
The captured face data is processed into features and labels.

### Face Recognition:
The script `face_recognition.py` uses the saved data to recognize faces in real-time.

## Setup Instructions

### 1. Prerequisites
Make sure you have the following installed:
- Python 3.7 or higher
- OpenCV: Install using `pip install opencv-python`
- NumPy: Install using `pip install numpy`

### 2. Clone the Repository
git clone https://ManishaBadhe//Face-Recognition-Project.git
cd Face-Recognition-Project

### 3. Download Haar Cascade File
Ensure the haarcascade_frontalface_alt.xml file is present in the project directory.

## How to Run the Project
#### 1. Data Collection
Run the script to collect face data for a user:
python face_data_collect.py
Enter the name of the user when prompted.
The webcam will open, and 50 images of the user’s face will be captured and saved.

#### 2. Face Recognition
Run the script to recognize faces:
python face_recognition.py
The webcam will open, and detected faces will be labeled with the user’s name.
