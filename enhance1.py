# Enhanced Blink Detection with MediaPipe (Preserving Core Logic)

import cv2
import time
import numpy as np
from collections import deque
from twilio.rest import Client
import mediapipe as mp

# Twilio Configuration
TWILIO_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""
RECIPIENT_PHONE_NUMBER = ""# The number to receive the voice call

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye landmark indices (MediaPipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

dynamic_EAR_threshold = 0.25
calibration_frames = 50
ear_samples = []
eye_closed_start_time = None

blink_count = 0
LONG_BLINK_DURATION = 3.0
TRACKING_WINDOW = 10
tracking_start_time = None

ear_buffer = deque(maxlen=5)

# EAR calculation
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def send_voice_alert(message):
    try:
        client.calls.create(
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            twiml=f'<Response><Say voice="alice" language="en-US">{message}</Say></Response>'
        )
        print(f"Voice Call initiated: {message}")
    except Exception as e:
        print(f"Error initiating voice call: {e}")

def extract_eye_coords(landmarks, eye_indices, img_w, img_h):
    return np.array([
        [int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)]
        for i in eye_indices
    ], dtype='float32')

def detect_blinks():
    global blink_count, tracking_start_time, dynamic_EAR_threshold, eye_closed_start_time

    cap = cv2.VideoCapture(0)
    calibrated = False
    print("Calibrating EAR threshold... Please keep your eyes open.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = extract_eye_coords(landmarks, LEFT_EYE_IDX, w, h)
            right_eye = extract_eye_coords(landmarks, RIGHT_EYE_IDX, w, h)

            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0
            ear_buffer.append(avg_EAR)
            smoothed_EAR = np.mean(ear_buffer)

            for pt in np.concatenate([left_eye, right_eye]):
                cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)

            if not calibrated:
                ear_samples.append(avg_EAR)
                if len(ear_samples) >= calibration_frames:
                    dynamic_EAR_threshold = np.mean(ear_samples) * 0.8
                    calibrated = True
                    print(f"Calibrated EAR threshold: {dynamic_EAR_threshold:.3f}")
                cv2.putText(frame, "Calibrating...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Blink Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Detect long blink
            if smoothed_EAR < dynamic_EAR_threshold:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                elif time.time() - eye_closed_start_time >= LONG_BLINK_DURATION:
                    blink_count += 1
                    eye_closed_start_time = None
                    tracking_start_time = time.time()
            else:
                eye_closed_start_time = None

            if tracking_start_time:
                if time.time() - tracking_start_time >= TRACKING_WINDOW:
                    if blink_count == 1:
                        send_voice_alert("EMERGENCY! Immediate attention needed.")
                    elif blink_count == 2:
                        send_voice_alert("Patient needs to use the restroom.")
                    elif blink_count == 3:
                        send_voice_alert("Patient needs water.")
                    elif blink_count == 4:
                        send_voice_alert("Patient needs food.")
                    else:
                        print("No command detected.")

                    blink_count = 0
                    tracking_start_time = None

            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_blinks()