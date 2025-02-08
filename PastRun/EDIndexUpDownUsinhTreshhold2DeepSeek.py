import cv2
import mediapipe as mp
import numpy as np

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

# Given data
min_max_scaled_data = {
    "indexUp.jpg": [0., 0.63424403, 0.99340856, 0.65059165, 0., 1., 1., 1., 1., 1., 0.94746419, 0.19886736, 0.07902441, 1., 0.77911549, 0.16199547, 0.07137699, 0.93715797, 0.60860068, 0.2168745, 0.13830083],
    "thumbUp.jpg": [0., 0., 0., 0.84539186, 1., 0.08593922, 0.17553055, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.16615251, 0.02233014, 0.01314123],
    "middleUp.jpg": [0., 1., 0.96961694, 1., 0.24036723, 0., 0.1285491, 0.07188041, 0.0519672, 0.33739682, 1., 1., 1., 0.40320996, 0.24732939, 0.09559927, 0.03036351, 0.48758967, 0.53512858, 0.22059179, 0.11664795],
    "rinkUp.jpg": [0., 0.6851529, 1., 0., 0.15158922, 0.33026348, 0., 0.00877957, 0.02079346, 0.37557257, 0.11670605, 0.22834288, 0.19747594, 0.37606732, 1., 1., 1., 0.21985286, 0., 0., 0.],
    "pinkyUp.jpg": [0., 0.62429938, 0.80152716, 0.14736532, 0.28273853, 0.13686549, 0.16248646, 0.11557564, 0.05786677, 0.34750098, 0.53411722, 0.18418036, 0.05774592, 0.6662434, 0.70516208, 0.1922978, 0.07106311, 1., 1., 1., 1.]
}

standardized_data = {
    "indexUp.jpg": [0., 0.13998086, 0.62722833, 0.31226835, -0.96706604, 1.90792707, 1.96975661, 1.98770878, 1.99704685, 1.81166077, 1.0398061, -0.35440184, -0.50468178, 1.53793506, 0.6344737, -0.35443729, -0.42533483, 1.04329758, 0.41731052, -0.20511711, -0.30574107],
    "thumbUp.jpg": [0., -1.81106806, -1.96362033, 0.81119388, 1.92022403, -0.62180285, -0.32829708, -0.62510918, -0.58353542, -1.26988797, -1.26305416, -0.92549434, -0.71701886, -1.47233604, -1.48898341, -0.80306893, -0.61137739, -1.35171454, -0.84195034, -0.73657465, -0.6375774],
    "middleUp.jpg": [0., 1.26511214, 0.56517885, 1.20717882, -0.27305612, -0.85964591, -0.45924903, -0.43729877, -0.44942979, -0.23018323, 1.16749708, 1.94623131, 1.96996208, -0.25856476, -0.8148942, -0.5383155, -0.53223556, -0.10562453, 0.20820006, -0.19496218, -0.36314947],
    "rinkUp.jpg": [0., 0.29658572, 0.64441907, -1.35403777, -0.529384, 0.05438237, -0.81755549, -0.60216977, -0.52987618, -0.11254279, -0.97939413, -0.26975624, -0.18640477, -0.34027147, 1.23648814, 1.96633951, 1.99510073, -0.78985594, -1.31484042, -0.79757629, -0.67241883],
    "pinkyUp.jpg": [0., 0.10938934, 0.12679407, -0.97660328, -0.15071788, -0.48086068, -0.36465501, -0.32313107, -0.43420546, -0.19904678, 0.0351451, -0.39657888, -0.56185667, 0.5332372, 0.43291577, -0.27051778, -0.42615295, 1.20389743, 1.53128017, 1.93423023, 1.97888677]
}

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

            # Find the closest finger
            closest_finger_min_max = find_closest_finger(min_max_scaled_distances, min_max_scaled_data)
            closest_finger_z_score = find_closest_finger(z_score_distances, standardized_data)

            # Display the result
            cv2.putText(frame, f"Min-Max: {closest_finger_min_max}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Z-Score: {closest_finger_z_score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()