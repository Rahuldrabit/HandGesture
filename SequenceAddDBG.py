import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Initialize ChromaDB (same as before)
chroma_client = chromadb.PersistentClient(path="chroma_storage")
try:
    collection = chroma_client.get_collection(name="personal_collection")
except:
    collection = chroma_client.create_collection(name="personal_collection")

# Initialize MediaPipe Hands (same as before)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Helper functions (same as before)
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def z_score_standardization(data, mean, std, eps=1e-8):
    if abs(std) < eps:
        return 0.0
    return (data - mean) / std

def find_closest_finger(distances, collection):
    min_distance = float('inf')
    closest_finger = None

    results = collection.get(include=["embeddings", "metadatas"])
    embeddings = results["embeddings"]
    metadatas = results["metadatas"]

    for i, embedding in enumerate(embeddings):
        distance = euclidean_distance(distances, embedding)
        if distance < min_distance:
            min_distance = distance
            closest_finger = metadatas[i]["finger"]
    return closest_finger

# Finger mapping dictionary
finger_map = {
    "Punch": "1", "TumbsIndexUp": "2", "TumbsUp": "3", "IndexUp": "4",
    "MiddleUp": "5", "RingUp": "6", "PinkyUp": "7"
}

# Initialize tracking variables
last_finger = None
start_time = None
finger_sequence = []
time_sequence = []
standardized_time_sequence = []
all_sequences = []

# Capture video
cap = cv2.VideoCapture(0)

def new_func(frame, sequence_text):
    cv2.putText(frame, sequence_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate distances and standardize (same as before)
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]
            distances = []
            for i in range(21):
                landmark = hand_landmarks.landmark[i]
                landmark_pos = [landmark.x, landmark.y, landmark.z]
                distances.append(euclidean_distance(wrist_pos, landmark_pos))

            mean = np.mean(distances)
            std = np.std(distances)
            z_score_distances = [float(z_score_standardization(d, mean, std)) for d in distances]

            closest_finger_name = find_closest_finger(z_score_distances, collection) or "Unknown"

            # Map finger name to number
            closest_finger = finger_map.get(closest_finger_name)
            if closest_finger is None:
                continue

            current_time = time.time()

            if last_finger is None:
                last_finger = closest_finger
                start_time = current_time
            elif closest_finger == last_finger:
                time_diff = current_time - start_time
            else:
                time_diff = current_time - start_time

                if time_diff > 0:  # Corrected logic here
                    finger_sequence.append(last_finger)
                    time_sequence.append(time_diff)

                    # Standardize time sequence *before* checking length
                    time_mean = np.mean(time_sequence)
                    time_std = np.std(time_sequence)
                    standardized_time_sequence = [z_score_standardization(t, time_mean, time_std) for t in time_sequence]

                    last_finger = closest_finger
                    start_time = current_time

                    if len(finger_sequence) == 6:
                        finger_time_pairs = list(zip(finger_sequence, standardized_time_sequence))
                        all_sequences.append(finger_time_pairs)
                        print(f"Finger : Time {finger_time_pairs}")  # Print here

                        finger_sequence = []  # Clear *after* processing
                        time_sequence = []
                        standardized_time_sequence = []

            # ... (Drawing landmarks and display)
            sequence_text = "Sequence: " + ", ".join([f"{f}({t:.1f}s)" for f, t in zip(finger_sequence, time_sequence)])

            new_func(frame, sequence_text)
            cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to JSON (same as before)
with open("finger_sequences.json", "w") as f:
    json.dump(all_sequences, f, indent=4)

print("Sequences saved to finger_sequences.json")