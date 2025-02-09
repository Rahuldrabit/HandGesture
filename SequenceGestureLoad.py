import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_storage")

# Check if Sequence_collection exists, otherwise create it
try:
    SequenceCollection = chroma_client.get_collection(name="Sequence_collection")
except Exception as e:
    print("Creating Sequence_collection because it does not exist:", e)
    SequenceCollection = chroma_client.create_collection(name="Sequence_collection")

# Check if personal_collection exists, otherwise create it
try:
    collection = chroma_client.get_collection(name="personal_collection")
except Exception as e:
    print("Creating personal_collection because it does not exist:", e)
    collection = chroma_client.create_collection(name="personal_collection")

# Load or initialize image_count
if os.path.exists("image_count.json"):
    with open("image_count.json", "r") as f:
        image_count = json.load(f).get("count", 0)
else:
    image_count = 0

Gesture_name = input("Enter the name of the finger: ")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance between two vectors
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to scale data using Z-score standardization
def z_score_standardization(data, mean, std, eps=1e-8):
    if abs(std) < eps:
        return 0.0  # Avoid division by zero
    return (data - mean) / std



# Function to find the closest finger from the personal collection using z-score distances
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

# Initialize tracking variables
last_finger = None
start_time = None
finger_sequence = []      # List of tuples: (finger name, duration)
time_sequence = []        # List of durations
Sc_dtime_sequence = []    # Normalized durations

# Variables for keeping track of the min and max time in the current sequence
MaxTime = 0
MinTime = 1

# Refresh interval (in seconds) for sequence popping and recalculation
refresh_interval = 0.5
last_refresh = time.time()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate distances from wrist (landmark 0) to all landmarks
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]

            distances = []
            for i in range(21):
                landmark = hand_landmarks.landmark[i]
                landmark_pos = [landmark.x, landmark.y, landmark.z]
                distances.append(euclidean_distance(wrist_pos, landmark_pos))

            # Standardize distances using Z-score
            mean = np.mean(distances)
            std = np.std(distances)
            z_score_distances = [float(z_score_standardization(d, mean, std)) for d in distances]

            # Find closest finger using the personal collection embeddings
            closest_finger = find_closest_finger(z_score_distances, collection) or "Unknown"

            # Update tracking variables:
            if last_finger is None:
                last_finger = closest_finger
                start_time = current_time
                time_diff = 0
            else:
                if closest_finger == last_finger:
                    time_diff = current_time - start_time
                else:
                    # When the finger changes, record the previous finger's duration (if any)
                    if time_diff > 0:
                        finger_sequence.append((last_finger, time_diff))
                        time_sequence.append(time_diff)
                    # Reset for the new finger
                    last_finger = closest_finger
                    start_time = current_time
                    time_diff = 0

            # Draw hand landmarks and their indices on the frame
            for i in range(21):
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.putText(frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display current finger and duration on the frame
            display_text = f"Finger: {last_finger} ({time_diff:.1f}s)"
            cv2.putText(frame, display_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the finger sequence (name and duration) on the frame
            sequence_text = "Sequence: " + ", ".join([f"{f}({t:.1f}s)" for f, t in finger_sequence])
            cv2.putText(frame, sequence_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Every refresh_interval seconds, pop older entries if more than 5 samples exist and recalc normalization
            if (current_time - last_refresh) > refresh_interval:
                if len(time_sequence) > 5:
                    popped_time = time_sequence.pop(0)
                    print(f"Popped from time_sequence: {popped_time}")
                if len(Sc_dtime_sequence) > 5:
                    popped_norm = Sc_dtime_sequence.pop(0)
                    print(f"Popped from Sc_dtime_sequence: {popped_norm}")
                if len(finger_sequence) > 5:
                    popped_finger = finger_sequence.pop(0)
                    print(f"Popped from finger_sequence: {popped_finger}")

                # Recalculate normalization if we have enough data points
                if len(time_sequence) > 5:
                    sqMean = np.mean(time_sequence)
                    sqStd = np.std(time_sequence)
                    Sc_dtime_sequence = [float(z_score_standardization(d, sqMean, sqStd)) for d in time_sequence]
                    MaxTime = max(time_sequence)
                    MinTime = min(time_sequence)
                else:
                    MaxTime = 0
                    MinTime = 0

                last_refresh = current_time  # Reset the refresh timer

            # Print sequences and statistics to the console
            print(sequence_text)
            print(f"MaxTime: {MaxTime}")
            print(f"MinTime: {MinTime}")
            print(f"time_sequence: {time_sequence}")
            print(f"Sc_dtime_sequence: {Sc_dtime_sequence}")

    # Show the frame
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
