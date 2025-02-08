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
except:
    SequenceCollection = chroma_client.create_collection(name="Sequence_collection")

# Check if collection exists, otherwise create it
try:
    collection = chroma_client.get_collection(name="personal_collection")
except:
    collection = chroma_client.create_collection(name="personal_collection")



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to scale data using Z-score standardization
def z_score_standardization(data, mean, std, eps=1e-8):
    if abs(std) < eps:
        return 0.0  # or return data - mean, depending on your needs
    return (data - mean) / std


# Function to find the closest finger
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
# List to store the sequence of fingers and their durations
finger_sequence = []
finger_sequence1 = []


MaxTime = 0
MinTime = 1
time_sequence = []
Sc_dtime_sequence = []
fingerTime = [] 


def FingerTimeEnum(finger_sequence, Sc_Time_sequence):
    # Ensure both sequences are of the same length
    if len(finger_sequence) != len(Sc_Time_sequence):
        raise ValueError("finger_sequence and Sc_Time_sequence must have the same length")

    FingerTimeSequence = []
    for finger, time in zip(finger_sequence, Sc_Time_sequence):
        FingerTimeSequence.append((finger, time))

    return FingerTimeSequence



# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate distances from wrist to all landmarks
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]

            distances = []
            for i in range(21):
                landmark = hand_landmarks.landmark[i]
                landmark_pos = [landmark.x, landmark.y, landmark.z]
                distances.append(euclidean_distance(wrist_pos, landmark_pos))

            # Standardize distances
            mean = np.mean(distances)
            std = np.std(distances)
            z_score_distances = [float(z_score_standardization(d, mean, std)) for d in distances]

            # Find closest finger
            closest_finger = find_closest_finger(z_score_distances, collection) or "Unknown"
            current_time = time.time()
            
            if closest_finger == "Punch":
                closest_finger = "1"
            elif closest_finger == "TumbsIndexUp":
                closest_finger = "2"
            elif closest_finger == "TumbsUp":
                closest_finger = "3"
            elif closest_finger == "IndexUp":
                closest_finger = "4"
            elif closest_finger == "MiddleUp":
                closest_finger = "5"
            elif closest_finger == "RingUp":
                closest_finger = "6"
            elif closest_finger == "PinkyUp":
                closest_finger = "7"
            else:
                continue


            # Update tracking variables
            if last_finger is None:
                last_finger = closest_finger
                start_time = current_time
                time_diff = 0
            else:
                if closest_finger == last_finger:
                    time_diff = current_time - start_time
                else:
                    
                    if time_diff > 0:
                        # Append the new finger and timestamp to the sequence
                        finger_sequence.append((last_finger, time_diff))
                        time_sequence.append(time_diff)
                    last_finger = closest_finger
                    start_time = current_time
                    time_diff = 0

                    # Keep only the last 5 fingers in the sequence
                    if len(finger_sequence) > 6:
                        fingerTime = FingerTimeEnum(finger_sequence, Sc_dtime_sequence)
                        # finger_sequence1 = finger_sequence[-6:]
                        finger_sequence = []

                    if len(time_sequence) > 6:
                        time_sequence = []


            # Draw landmarks
            for i in range(21):
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.putText(frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display recognition results
            display_text = f"Finger: {last_finger} ({time_diff:.1f}s)"
            cv2.putText(frame, display_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the finger sequence
            sequence_text = "Sequence: " + ", ".join([f"{finger[0]}({finger[1]:.1f}s)" for finger in finger_sequence])

            # fingerTime = FingerTimeEnum(finger_sequence, Sc_dtime_sequence)

            print(f"Finger : Time {fingerTime}")

            if len(time_sequence) > 3:
                sqMean = np.mean(time_sequence)
                sqStd = np.std(time_sequence)

                Sc_dtime_sequence = [float(z_score_standardization(d, sqMean, sqStd)) for d in time_sequence]  
                MaxTime = max(time_sequence)
                MinTime = min(time_sequence)
                if len(Sc_dtime_sequence) > 6:
                    Sc_dtime_sequence = []
                

            cv2.putText(frame, sequence_text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(sequence_text)
            print(f"time_sequence: {time_sequence}")
            print(f"Sc_dtime_sequence: {Sc_dtime_sequence}")

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()