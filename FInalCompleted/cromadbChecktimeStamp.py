import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_storage")

# Create or get collection with proper exception handling
try:
    collection = chroma_client.get_collection(name="personal_collection")
except Exception as e:
    collection = chroma_client.create_collection(name="personal_collection")

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

            # Update tracking variables
            if last_finger is None:
                last_finger = closest_finger
                start_time = current_time
                time_diff = 0
            else:
                if closest_finger == last_finger:
                    time_diff = current_time - start_time
                else:
                    last_finger = closest_finger
                    start_time = current_time
                    time_diff = 0

            # Draw landmarks
            for i in range(21):
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.putText(frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display recognition results
            display_text = f"Finger: {last_finger} ({time_diff:.1f}s)"
            cv2.putText(frame, display_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()