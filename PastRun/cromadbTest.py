import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="personal_collection")


# python take imput into a variable from user
finger_name = input("Enter the name of the finger: ")

image_count = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to scale data using Min-Max scaling
def min_max_scaling(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

# Function to scale data using Z-score standardization
def z_score_standardization(data, mean, std):
    return (data - mean) / std



# Function to find the closest finger
def find_closest_finger(distances, data):
    min_distance = float('inf')
    closest_finger = None
    for finger, values in data.items():
        distance = euclidean_distance(distances, values)
        if distance < min_distance:
            min_distance = distance
            closest_finger = finger
    return closest_finger

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist position (landmark 0)
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]

            # Calculate distances from wrist to other keypoints
            distances = []
            for i in range(21):
                landmark = hand_landmarks.landmark[i]
                landmark_pos = [landmark.x, landmark.y, landmark.z]
                distances.append(euclidean_distance(wrist_pos, landmark_pos))

            # Min-Max scaling
            min_val = min(distances)
            max_val = max(distances)
            min_max_scaled_distances = [min_max_scaling(d, min_val, max_val) for d in distances]

            # Z-score standardization
            mean = np.mean(distances)
            std = np.std(distances)
            z_score_distances = [z_score_standardization(d, mean, std) for d in distances]


            # store z_score_distances in the database
            collection.add(
                embeddings=[z_score_distances],
                metadatas=[{"finger": finger_name}],
                ids=[f"{finger_name}_{image_count}"]
            )
            image_count += 1

            time.sleep(0.9)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()