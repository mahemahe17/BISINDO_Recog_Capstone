import cv2
import time
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from collections import deque
import pyttsx3

STATIC_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
                 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DYNAMIC_LABELS = ['AKU', 'HALO', 'APA', 'KABAR', 'KAMU', 'NAMA', 'PERKENALKAN', 
                  'SIAPA', 'TERIMA KASIH', 'TOLONG', 'SALAM KENAL', 'MAAF', 
                  'BERASAL', 'DARI', 'MANA', 'SAMA-SAMA', 'YA', 'TIDAK', 'JAKARTA']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results, include_pose=True):
    pose = [[res.x, res.y] for res in results.pose_landmarks.landmark] if results.pose_landmarks else [[0, 0]] * 33
    lh = [[res.x, res.y] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [[0, 0]] * 21
    rh = [[res.x, res.y] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [[0, 0]] * 21

    if include_pose:
        return np.array(pose + lh + rh).flatten()
    else:
        return np.array(lh + rh).flatten()
    
def mediapipe_detection(image, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results, include_pose=True):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if include_pose:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 2)
    min_val = keypoints.min(axis=0)
    max_val = keypoints.max(axis=0)
    range_val = np.where((max_val - min_val) == 0, 1, (max_val - min_val))
    normed = (keypoints - min_val) / range_val
    return normed.flatten()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def compute_movement(curr, prev):
    if prev is None:
        return 0.0
    return np.linalg.norm(curr - prev)

model_static = tf.keras.models.load_model("static_model.h5")
model_dynamic = tf.keras.models.load_model("high_dynamic_model.h5")

SEQUENCE_LENGTH = 30
MOVEMENT_THRESHOLD = 0.04
CONFIDENCE_THRESHOLD = 0.9
silence_timeout = 2.0

sequence = []
prev_keypoints = None
movement_history = deque(maxlen=5)
frame_counter = 0
last_spoken = ""
last_speak_time = 0
history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results, include_pose=True)

        # Cek apakah tangan terdeteksi
        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            cv2.putText(image, 'No hand detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('BISINDO Gesture Recognition', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            continue

        # Ekstraksi dan normalisasi keypoint
        keypoints_static = extract_keypoints(results, include_pose=False)
        keypoints_dynamic = extract_keypoints(results, include_pose=True)
        norm_static = normalize_keypoints(keypoints_static)
        norm_dynamic = normalize_keypoints(keypoints_dynamic)

        # Hitung pergerakan
        movement = compute_movement(norm_dynamic, prev_keypoints)
        movement_history.append(movement)
        prev_keypoints = norm_dynamic
        is_dynamic = np.mean(movement_history) > MOVEMENT_THRESHOLD

        # Inisialisasi hasil
        final_pred = "-"
        final_conf = 0.0
        label_static, conf_static = "-", 0.0
        label_dynamic, conf_dynamic = "-", 0.0

        # ======= PREDIKSI =========
        if is_dynamic:
            sequence.append(norm_dynamic)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[-SEQUENCE_LENGTH:]
            if len(sequence) == SEQUENCE_LENGTH and frame_counter % 10 == 0:
                input_seq = np.expand_dims(np.array(sequence), axis=0)
                pred_dynamic = model_dynamic.predict(input_seq, verbose=0)[0]
                conf_dynamic = np.max(pred_dynamic)
                if conf_dynamic > CONFIDENCE_THRESHOLD:
                    label_dynamic = DYNAMIC_LABELS[np.argmax(pred_dynamic)]
                    final_pred = label_dynamic
                    final_conf = conf_dynamic
                    history.append(label_dynamic)
                    if len(history) == history.maxlen:
                        final_pred = max(set(history), key=history.count)
                        history.clear()
        else:
            pred_static = model_static.predict(np.expand_dims(norm_static, axis=0), verbose=0)[0]
            conf_static = np.max(pred_static)
            if conf_static > CONFIDENCE_THRESHOLD:
                label_static = STATIC_LABELS[np.argmax(pred_static)]
                final_pred = label_static
                final_conf = conf_static

        # === Bicara jika gesture berbeda dan timeout terpenuhi
        if final_pred != "-" and final_pred != last_spoken:
            if time.time() - last_speak_time > silence_timeout:
                speak(final_pred)
                last_spoken = final_pred
                last_speak_time = time.time()

        # === UI Tampilan
        cv2.rectangle(image, (0, 0), (640, 90), (245, 245, 245), -1)
        cv2.putText(image, f'STATIC: {label_static} ({conf_static:.2f})', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f'DYNAMIC: {label_dynamic} ({conf_dynamic:.2f})', (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        cv2.putText(image, f'FINAL: {final_pred} ({final_conf:.2f})', (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)

        cv2.imshow('BISINDO Gesture Recognition', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()