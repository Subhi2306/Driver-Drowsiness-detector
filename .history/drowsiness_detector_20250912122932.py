# Importing Libraries
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

# Alarm control
alarm_on = False

# Function to compute distance between points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to compute blink ratio (EAR)
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > EAR_DROWSY_THRESHOLD:
        return 2  # Active
    elif EAR_SLEEP_THRESHOLD < ratio <= EAR_DROWSY_THRESHOLD:
        return 1  # Drowsy
    else:
        return 0  # Sleeping

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

        # Compute EAR for both eyes
        left_ear = compute(landmarks[37], landmarks[41]) + compute(landmarks[38], landmarks[40])
        right_ear = compute(landmarks[43], landmarks[47]) + compute(landmarks[44], landmarks[46])
        horiz_left = compute(landmarks[36], landmarks[39])
        horiz_right = compute(landmarks[42], landmarks[45])
        left_ratio = left_ear / (2.0 * horiz_left)
        right_ratio = right_ear / (2.0 * horiz_right)
        ear_values.append((left_ratio + right_ratio)/2)

    # Optional: show frame during calibration
    cv2.putText(frame, "Calibrating...", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

cv2.destroyWindow("Calibration")
EAR_OPEN_AVG = np.mean(ear_values)
EAR_SLEEP_THRESHOLD = EAR_OPEN_AVG * 0.75
EAR_DROWSY_THRESHOLD = EAR_OPEN_AVG * 0.85
print(f"Calibration done. EAR thresholds: Sleep={EAR_SLEEP_THRESHOLD:.2f}, Drowsy={EAR_DROWSY_THRESHOLD:.2f}")

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

        # Calculate blink status using calibrated thresholds
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()

        elif left_blink == 1 or right_blink == 1:
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
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                alarm_on = False  # Stop alarm

        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        alarm_on = False
        break

cap.release()
cv2.destroyAllWindows()
