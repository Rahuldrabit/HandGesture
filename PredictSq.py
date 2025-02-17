import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_storage")
try:
    collection = chroma_client.get_collection(name="personal_collection")
except Exception as e:
    print("Collection not found, creating new one.")
    collection = chroma_client.create_collection(name="personal_collection")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Utility functions
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def z_score_standardization(data, eps=1e-8):
    mean = np.mean(data)
    std = np.std(data)
    return [(x - mean) / (std + eps) for x in data]

def find_closest_finger(distances, collection):
    results = collection.get(include=["embeddings", "metadatas"])
    min_distance = float('inf')
    closest_finger = None
    for i, embedding in enumerate(results["embeddings"]):
        distance = euclidean_distance(distances, embedding)
        if distance < min_distance:
            min_distance = distance
            closest_finger = results["metadatas"][i]["finger"]
    return closest_finger

# Load model and encoder
model = load_model('lstm_finger_model.h5')
label_encoder = joblib.load('label_encoder.pkl')  # Must match training labels

# Fixed finger mapping (use integers)
finger_map = {
    "Punch": 1,
    "Punc": 1,
    "TumbsIndexUp": 2,
    "TumbsUp": 3,
    "IndexUp": 4,
    "MiddleUp": 5,
    "Middle": 5,
    "RingUp": 6,
    "PinkyUp": 7,
    "AllUp": 8
}

# Reverse mapping for display
label_to_gesture = {
    1: "Punch",
    2: "Thumbs+Index",
    3: "Thumbs Up",
    4: "Index Up",
    5: "Middle Up",
    6: "Ring Up",
    7: "Pinky Up",
    8: "All Fingers Up"
}

def process_sequence(sequence):
    sequence = np.array(sequence, dtype=np.float32).reshape(1, 4, 2)
    pred_probs = model.predict(sequence)[0]
    predicted_index = np.argmax(pred_probs)
    return label_encoder.inverse_transform([predicted_index])[0]

# Tracking variables
last_finger = None
start_time = None
finger_sequence = []
time_sequence = []
prediction = ""
recording = True

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process hand landmarks
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    current_finger = None

    if recording and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Landmark processing...
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]
            distances = [euclidean_distance(wrist_pos, [lm.x, lm.y, lm.z]) 
                        for lm in hand_landmarks.landmark]
            
            # Z-score normalization
            z_scores = z_score_standardization(distances)
            closest = find_closest_finger(z_scores, collection)
            current_finger = finger_map.get(closest, None)

    # Sequence collection logic
    if recording and current_finger is not None:
        current_time = time.time()
        
        if last_finger is None:
            # Start new sequence
            last_finger = current_finger
            start_time = current_time
        elif current_finger != last_finger:
            # Record time difference
            time_diff = current_time - start_time
            finger_sequence.append(last_finger)
            time_sequence.append(time_diff)
            
            # Update tracking
            last_finger = current_finger
            start_time = current_time

            # When we have 4 gestures
            if len(finger_sequence) == 4:
                # Process sequence
                standardized_times = z_score_standardization(time_sequence)
                sequence = list(zip(finger_sequence, standardized_times))
                
                try:
                    np_array = np.array(sequence, dtype=np.float32).reshape(1, 4, 2)
                    prediction_sequence = np.argmax(model.predict(np_array), axis=1)
                    print(f"Predicted sequence: {prediction_sequence}")

                    # Predict final gesture
                    predicted_label_index = np.argmax(prediction_sequence) 

                    # Use the label encoder to convert the index back to the original label
                    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]  

                    print(f"Predicted Gesture: {predicted_label}")

                    pred_label = process_sequence(sequence)
                    prediction = label_to_gesture.get(pred_label, "Unknown")
                    print(f"Predicted: {prediction}")
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    prediction = "Error"

                # Reset for next sequence
                finger_sequence = []
                time_sequence = []
                last_finger = None
                recording = False

    # Display UI
    cv2.putText(frame, f"Status: {'Recording' if recording else 'Paused'}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if prediction:
        cv2.putText(frame, f"Prediction: {prediction}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Hand Gesture Recognition', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space to toggle recording
        recording = not recording
        prediction = ""
        finger_sequence = []
        time_sequence = []
        last_finger = None

cap.release()
cv2.destroyAllWindows()