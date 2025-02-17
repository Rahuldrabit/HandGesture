import chromadb
import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model  # Use TensorFlow's Keras
from keras.losses import sparse_categorical_crossentropy, mean_squared_error
from keras.metrics import Accuracy


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

def custom_pad_sequences(finger_seqs, time_seqs, pad_value=0, maxlen=10):
    padded_fingers = []
    padded_times = []
    for f_seq, t_seq in zip(finger_seqs, time_seqs):
        # Pad fingers
        if len(f_seq) > maxlen:
            padded_f = f_seq[:maxlen]
        else:
            padded_f = f_seq + [pad_value] * (maxlen - len(f_seq))
        padded_fingers.append(padded_f)
        
        # Pad times
        if len(t_seq) > maxlen:
            padded_t = t_seq[:maxlen]
        else:
            padded_t = t_seq + [pad_value] * (maxlen - len(t_seq))
        padded_times.append(padded_t)
    return np.array(padded_fingers), np.array(padded_times)

# Define custom objects for model loading
custom_objects = {
    'sparse_categorical_crossentropy': sparse_categorical_crossentropy,
    'mse': mean_squared_error,
    'accuracy': Accuracy()  # Use an instance of Accuracy
}

# Load the model with custom objects
model = load_model('lstm_finger_Sq_model.h5', custom_objects=custom_objects)

# Fixed finger mapping
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
            wrist = hand_landmarks.landmark[0]
            wrist_pos = [wrist.x, wrist.y, wrist.z]
            distances = [euclidean_distance(wrist_pos, [lm.x, lm.y, lm.z]) 
                         for lm in hand_landmarks.landmark]
            
            z_scores = z_score_standardization(distances)
            closest = find_closest_finger(z_scores, collection)
            current_finger = finger_map.get(closest, None)

    # Sequence collection logic
    if recording and current_finger is not None:
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

            if len(finger_sequence) == 7:  # Collect 6 elements
                standardized_times = z_score_standardization(time_sequence)
                
                # Apply padding
                new_padded_fingers, new_padded_times = custom_pad_sequences(
                    [finger_sequence], [standardized_times], 
                    pad_value=0, maxlen=10
                )
                
                # Predict
                pred_fingers, pred_times = model.predict(
                    [new_padded_fingers, new_padded_times]
                )
                
                # Decode predictions
                predicted_fingers = np.argmax(pred_fingers, axis=-1)[0]
                predicted_gestures = [label_to_gesture.get(int(f), "Unknown") 
                                      for f in predicted_fingers if f != 0]
                
                prediction = " -> ".join(predicted_gestures)
                print(f"Predicted Sequence: {prediction}")

                # Reset sequences and recording status
                finger_sequence = []
                time_sequence = []
                last_finger = None
                prediction = ""

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
    elif key == ord(' '):
        recording = not recording
        prediction = ""
        finger_sequence = []
        time_sequence = []
        last_finger = None

cap.release()
cv2.destroyAllWindows()
