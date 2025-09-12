# Import Libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound
import threading
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status counters
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alarm_on = False

# -----------------------------
# Helper Functions
# -----------------------------
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# EAR - Eye Aspect Ratio
def compute_ear(landmarks, left=True):
    if left:
        a = compute(landmarks[37], landmarks[41])
        b = compute(landmarks[38], landmarks[40])
        c = compute(landmarks[36], landmarks[39])
    else:
        a = compute(landmarks[43], landmarks[47])
        b = compute(landmarks[44], landmarks[46])
        c = compute(landmarks[42], landmarks[45])
    return (a + b) / (2.0 * c)

# MAR - Mouth Aspect Ratio
def compute_mar(landmarks):
    a = compute(landmarks[61], landmarks[67])
    b = compute(landmarks[62], landmarks[66])
    c = compute(landmarks[63], landmarks[65])
    d = compute(landmarks[60], landmarks[64])
    return (a + b + c) / (3.0 * d)

# Function to play continuous alarm
def play_alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(2500, 1000)  # frequency=2500Hz, duration=1s

# -----------------------------
# 1. Calibration Phase (5 sec)
# -----------------------------
print("Calibration: Keep your eyes open normally for 5 seconds...")
ear_values = []

start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_ear = compute_ear(landmarks, left=True)
        right_ear = compute_ear(landmarks, left=False)
        ear_values.append((left_ear + right_ear)/2)

    # Show calibration message
    cv2.putText(frame, "Calibrating...", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

cv2.destroyWindow("Calibration")
EAR_OPEN_AVG = np.mean(ear_values)
EAR_SLEEP_THRESHOLD = EAR_OPEN_AVG * 0.75
EAR_DROWSY_THRESHOLD = EAR_OPEN_AVG * 0.85
MAR_YAWN_THRESHOLD = 0.6  # You can tune this
print(f"Calibration done. EAR sleep={EAR_SLEEP_THRESHOLD:.2f}, drowsy={EAR_DROWSY_THRESHOLD:.2f}")

# -----------------------------
# 2. Main Detection Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Compute EAR
        left_ear = compute_ear(landmarks, left=True)
        right_ear = compute_ear(landmarks, left=False)
        ear_avg = (left_ear + right_ear)/2

        # Compute MAR for yawning
        mar = compute_mar(landmarks)

        # Head Pose Approximation using vertical distance between nose tip & chin
        head_tilt = compute(landmarks[33], landmarks[8])  # nose tip to chin

        # Decision Logic
        if ear_avg < EAR_SLEEP_THRESHOLD or mar > MAR_YAWN_THRESHOLD or head_tilt < 0.9*(EAR_OPEN_AVG*3):
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()

        elif ear_avg < EAR_DROWSY_THRESHOLD or mar > 0.5:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()

        else:
            sleep = 0
            drowsy = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                alarm_on = False

        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Enhanced Drowsiness Detector", frame)
    key = cv2.waitKey(1)
    if key == 27:
        alarm_on = False
        break

cap.release()
cv2.destroyAllWindows()
