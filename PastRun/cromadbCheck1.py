import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_storage")

# Check if collection exists, otherwise create it
try:
    collection = chroma_client.get_collection(name="personal_collection")
except:
    collection = chroma_client.create_collection(name="personal_collection")

# Load or initialize image_count
if os.path.exists("image_count.json"):
    with open("image_count.json", "r") as f:
        image_count = json.load(f).get("count", 0)
else:
    image_count = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to scale data using Z-score standardization
def z_score_standardization(data, mean, std):
    return (data - mean) / std

# Function to find the closest finger
def find_closest_finger(distances, collection):
    min_distance = float('inf')
    closest_finger = None

    # Query all embeddings from the collection
    results = collection.get(include=["embeddings", "metadatas"])
    embeddings = results["embeddings"]
    metadatas = results["metadatas"]

    # Compare with stored embeddings
    for i, embedding in enumerate(embeddings):
        distance = euclidean_distance(distances, embedding)
        if distance < min_distance:
            min_distance = distance
            closest_finger = metadatas[i]["finger"]

    return closest_finger

# Capture video from webcam
cap = cv2.VideoCapture(0)
last_capture_time = 0  # To control capture rate

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        # Capture at most 1 frame per second (adjustable)
        if current_time - last_capture_time >= 1.0:
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
                z_score_distances = [float(z_score_standardization(d, mean, std)) for d in distances]

                # Find the closest finger
                closest_finger = find_closest_finger(z_score_distances, collection)
                print(f"Detected finger: {closest_finger}")

                # Display the detected finger on the frame
                cv2.putText(frame, f"Finger: {closest_finger}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                last_capture_time = current_time  # Update last capture time

    # Display the frame
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()