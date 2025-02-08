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

# Initialize Kalman filters for each of the 21 hand keypoints
kalman_filters = []
for _ in range(21):
    kf = cv2.KalmanFilter(6, 3)  # 6 states (x,y,z + velocity), 3 measurements (x,y,z)
    dt = 1.0  # Time step (adjust based on expected frame rate)
    
    # State transition matrix (constant velocity model)
    kf.transitionMatrix = np.array([
        [1,0,0,dt,0,0],
        [0,1,0,0,dt,0],
        [0,0,1,0,0,dt],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], dtype=np.float32)
    
    # Measurement matrix
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0]
    ], dtype=np.float32)
    
    # Covariance matrices (tune these parameters)
    kf.processNoiseCov = 1e-4 * np.eye(6, dtype=np.float32)
    kf.measurementNoiseCov = 1e-2 * np.eye(3, dtype=np.float32)
    kf.errorCovPost = 1e-1 * np.eye(6, dtype=np.float32)
    
    kalman_filters.append(kf)

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
finger_sequence = []  # Stores sequence of fingers with durations

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Apply Kalman filtering to each landmark
            for i in range(21):
                kf = kalman_filters[i]
                landmark = hand_landmarks.landmark[i]
                
                # Get measurement
                meas = np.array([[landmark.x], [landmark.y], [landmark.z]], dtype=np.float32)
                
                # Kalman predict
                prediction = kf.predict()
                
                # Kalman correct
                kf.correct(meas)
                
                # Update landmark with smoothed coordinates
                landmark.x = kf.statePost[0, 0]
                landmark.y = kf.statePost[1, 0]
                landmark.z = kf.statePost[2, 0]

            # Calculate distances from wrist to all landmarks (using smoothed coordinates)
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
                    finger_sequence.append((last_finger, time_diff))
                    last_finger = closest_finger
                    start_time = current_time
                    time_diff = 0

                    # Keep only last 5 elements
                    if len(finger_sequence) > 5:
                        finger_sequence.pop(0)

            # Draw landmarks (smoothed positions)
            for i in range(21):
                x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                y = int(hand_landmarks.landmark[i].y * frame.shape[0])
                cv2.putText(frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display recognition results
            display_text = f"Finger: {last_finger} ({time_diff:.1f}s)"
            cv2.putText(frame, display_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the finger sequence
            sequence_text = "Sequence: " + ", ".join([f"{finger[0][0]}({finger[1]:.1f}s)" for finger in finger_sequence])
            cv2.putText(frame, sequence_text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()