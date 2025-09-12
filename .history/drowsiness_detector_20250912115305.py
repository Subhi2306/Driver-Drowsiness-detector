import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# ---------- Step 1: Eye Aspect Ratio Function ----------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ---------- Step 2: Initial Variables ----------
EYE_AR_THRESH = 0.25   # default (will update after calibration)
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

# Calibration variables
calibration_frames = 50
ear_samples = []
frame_count = 0
calibrated = False

# ---------- Step 3: Load Detector & Predictor ----------
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)  # Left eye landmarks
(rStart, rEnd) = (36, 42)  # Right eye landmarks

# ---------- Step 4: Start Video ----------
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # ---------- Calibration ----------
        if not calibrated:
            if frame_count < calibration_frames:
                ear_samples.append(ear)
                frame_count += 1
                cv2.putText(frame, f"Calibrating... {frame_count}/{calibration_frames}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                EYE_AR_THRESH = np.mean(ear_samples) * 0.75
                calibrated = True
                print("Calibrated EAR Threshold:", EYE_AR_THRESH)
        else:
            # ---------- Drowsiness Detection ----------
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                COUNTER = 0

        # Draw eyes
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # ---------- Show Result ----------
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
