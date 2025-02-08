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
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Helper Functions
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def min_max_normalization(data, data_min, data_max, eps=1e-8):
    if abs(data_max - data_min) < eps:
        return 0.0
    return (data - data_min) / (data_max - data_min)

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

finger_map = {
    "Punch": "1",
    "TumbsIndexUp": "2",
    "TumbsUp": "3",
    "IndexUp": "4",
    "MiddleUp": "5",
    "RingUp": "6",
    "PinkyUp": "7"
}

last_finger = None
start_time = None
finger_sequence = []
time_sequence = []
all_sequences = []

sequence_name = input("Input sequence name (avoid using 'p' or 'r'): ")

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

    # Process frames even when paused to show live video
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    current_finger = None

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

    # Update finger sequence only when recording
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

            if len(finger_sequence) == 6:
                # Normalize times
                t_min = min(time_sequence)
                t_max = max(time_sequence)
                normalized_times = [min_max_normalization(t, t_min, t_max) for t in time_sequence]
                sequence = list(zip(finger_sequence, normalized_times))
                all_sequences.append(sequence)
                print(f"Stored sequence: {sequence}")
                # Reset
                finger_sequence = []
                time_sequence = []
                last_finger = None
                recording = False  # Auto-pause after 6 gestures

    # Display UI
    state_text = f"Recording: {len(finger_sequence)}/6" if recording else "Paused"
    cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Hand Tracking', frame)

    # Key handling (non-blocking)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if not recording:
            recording = True
            print("Recording resumed. Show next angle.")
    elif key == ord('p'):
        recording = False
        print("Paused. Adjust angle and press 'r' to record next sequence.")

cap.release()
cv2.destroyAllWindows()

# Save data
if all_sequences:
    data_to_save = {sequence_name: all_sequences}
    with open("finger_sequences.json", "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Saved {len(all_sequences)} sequences under '{sequence_name}'")
else:
    print("No sequences recorded.")