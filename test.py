import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Ensure data files exist
if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
    print("Error: Missing data files. Ensure 'names.pkl' and 'faces_data.pkl' exist.")
    exit()

# Load face recognition data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure FACES and LABELS match in size
if isinstance(FACES, np.ndarray):
    faces_count = FACES.shape[0]
else:
    print("Error: FACES data is not in the expected numpy array format.")
    exit()

labels_count = len(LABELS)

if faces_count != labels_count:
    print(f"Warning: Mismatch in data sizes. Faces: {faces_count}, Labels: {labels_count}")
    min_length = min(faces_count, labels_count)
    FACES = FACES[:min_length]  # Trim FACES to match labels
    LABELS = LABELS[:min_length]  # Trim LABELS to match faces

print(f"Shape of Faces matrix --> {FACES.shape}")
print(f"Length of Labels list --> {len(LABELS)}")

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load UI background
imgBackground = cv2.imread("background.png")
if imgBackground is None:
    print("Error: 'background.png' file not found.")
    exit()

COL_NAMES = ['NAME', 'TIME']
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y+h, x:x+w]  # Use grayscale
        resized_img = cv2.resize(crop_img, (50, 87)).flatten().reshape(1, -1)  # Match 4350 features

        if resized_img.shape[1] != FACES.shape[1]:
            print(f"Feature mismatch: Expected {FACES.shape[1]}, got {resized_img.shape[1]}")
            continue

        output = knn.predict(resized_img)

        # Timestamp for attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        attendance_file = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(attendance_file)

        # Draw bounding boxes and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)

    key = cv2.waitKey(1)

    if key == ord('o'):
        speak("Attendance Taken..")
        time.sleep(1)

        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        with open(attendance_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
