import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize scalers
z_score_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Dictionary to store keypoints
keypoints_dict = {}

# Function to scale keypoints
def scale_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)
    z_score_scaled = z_score_scaler.fit_transform(keypoints)
    min_max_scaled = min_max_scaler.fit_transform(keypoints)
    robust_scaled = robust_scaler.fit_transform(keypoints)
    return z_score_scaled, min_max_scaled, robust_scaled

# Timestamp interval
interval = 0.3  # seconds
last_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(image_rgb)

    # Check if hand keypoints are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])

            # Scale the keypoints
            z_score_scaled, min_max_scaled, robust_scaled = scale_keypoints(keypoints)

            # Get current timestamp
            current_time = time.time()

            # Store keypoints every 0.3 seconds
            if current_time - last_time >= interval:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                keypoints_dict[timestamp] = {
                    'original': keypoints,
                    'z_score_scaled': z_score_scaled.tolist(),
                    'min_max_scaled': min_max_scaled.tolist(),
                    'robust_scaled': robust_scaled.tolist()
                }
                last_time = current_time

    # Display the frame
    cv2.imshow('Hand Keypoints', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Print the stored keypoints
for timestamp, keypoints in keypoints_dict.items():
    print(f"Timestamp: {timestamp}")
    print(f"Original Keypoints: {keypoints['original']}")
    print(f"Z-Score Scaled Keypoints: {keypoints['z_score_scaled']}")
    print(f"Min-Max Scaled Keypoints: {keypoints['min_max_scaled']}")
    print(f"Robust Scaled Keypoints: {keypoints['robust_scaled']}")
    print("-" * 40)