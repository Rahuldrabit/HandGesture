import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_storage")
try:
    collection = chroma_client.get_collection(name="personal_collection")
except Exception as e:
    print("Collection not found, creating a new one.")
    collection = chroma_client.create_collection(name="personal_collection")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Utility functions
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def z_score_standardization(data, eps=1e-8):
    mean = np.mean(data)
    std = np.std(data)
    if std < eps:
        return [0.0] * len(data)
    return [(x - mean) / std for x in data]

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

# Save Sequence Function
file_path = "finger_sequences.json"

def save_sequence(sequence_name, all_sequences):
    if not all_sequences:
        print("No sequences to save.")
        return

    # Read existing JSON data if file exists, else start with an empty dictionary
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = {}
    else:
        existing_data = {}

    if not isinstance(existing_data, dict):
        existing_data = {}

    # Append new data to the existing data
    if sequence_name in existing_data:
        existing_data[sequence_name].extend(all_sequences)
    else:
        existing_data[sequence_name] = all_sequences

    # Save updated data back to the JSON file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Saved {len(all_sequences)} sequences under '{sequence_name}'")

# Finger Mapping Dictionary
finger_map = {
    "Punch": "1",
    "Punc": "1",
    "TumbsIndexUp": "2",
    "TumbsUp": "3",
    "IndexUp": "4",
    "MiddleUp": "5",
    "Middle" : "5",
    "RingUp": "6",
    "PinkyUp": "7",
    "AllUp": "8"
}

# Tracking Variables
last_finger = None
start_time = None
finger_sequence = []
time_sequence = []
all_sequences = []

# Ask user for sequence name
sequence_name = input("Input sequence name (avoid using 'p' or 'r'): ")

# Video Capture and Control Variables
cap = cv2.VideoCapture(0)
recording = False  # Start paused

print("\nInstructions:")
print("  - Press 'r' to start/resume recording.")
print("  - Press 'p' to pause recording.")
print("  - Press 'q' to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame and hand landmarks
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    current_finger = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Only process finger detection if recording is active
    if recording and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]
            distances = []
            for i in range(21):
                landmark = hand_landmarks.landmark[i]
                landmark_pos = [landmark.x, landmark.y, landmark.z]
                distances.append(euclidean_distance(wrist_pos, landmark_pos))
            mean = np.mean(distances)
            std = np.std(distances)
            z_score_distances = [float((d - mean) / (std + 1e-8)) for d in distances]
            closest_finger_name = find_closest_finger(z_score_distances, collection)
            current_finger = finger_map.get(closest_finger_name, None)
            if closest_finger_name:
                cv2.putText(frame, f"Current Finger: {closest_finger_name}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Process only the first detected hand
            break

    # If recording and a valid finger is detected, update the sequence
    if recording and current_finger:
        current_time = time.time()
        if last_finger is None:
            last_finger = current_finger
            start_time = current_time
        elif current_finger != last_finger:
            time_diff = current_time - start_time
            finger_sequence.append(last_finger)
            time_sequence.append(time_diff)
            last_finger = current_finger
            start_time = current_time

    # Handle key presses (only one cv2.waitKey call per frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # If quitting and a sequence is in progress, finalize it first.
        if recording and last_finger is not None:
            current_time = time.time()
            time_diff = current_time - start_time
            finger_sequence.append(last_finger)
            time_sequence.append(time_diff)
            if finger_sequence:
                standardized_times = z_score_standardization(time_sequence)
                sequence = list(zip(finger_sequence, standardized_times))
                all_sequences.append(sequence)
                print(f"Stored sequence: {sequence}")
        break
    elif key == ord('r'):
        if not recording:
            recording = True
            # Reset sequence data when starting a new recording
            last_finger = None
            start_time = None
            finger_sequence = []
            time_sequence = []
            print("Recording resumed. Show next angle.")
    elif key == ord('p'):
        if recording:
            # Finalize the current sequence if possible
            if last_finger is not None:
                current_time = time.time()
                time_diff = current_time - start_time
                finger_sequence.append(last_finger)
                time_sequence.append(time_diff)
            if finger_sequence:
                standardized_times = z_score_standardization(time_sequence)
                sequence = list(zip(finger_sequence, standardized_times))
                all_sequences.append(sequence)
                print(f"Stored sequence: {sequence}")
            else:
                print("No sequence to store.")
            recording = False
            last_finger = None
            start_time = None
            finger_sequence = []
            time_sequence = []
            print("Paused. Adjust angle and press 'r' to record next sequence.")

    state_text = f"Recording: {len(finger_sequence)}" if recording else "Paused"
    cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Hand Tracking', frame)

cap.release()
cv2.destroyAllWindows()

# Save sequences to JSON file
save_sequence(sequence_name, all_sequences)
print("All sequences saved successfully.")
