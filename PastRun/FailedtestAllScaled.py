import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to find the closest finger
def find_closest_finger(distances, data):
    min_distance = float('inf')
    closest_finger = None
    for finger, values in data.items():
        # Compare with the first set of keypoints (21 values)
        distance = euclidean_distance(distances, values[0])
        if distance < min_distance:
            min_distance = distance
            closest_finger = finger
    return closest_finger

# Given keypoints data
image_keypoints_dict = {
    "index.jpg": [[0.0, 297.658812138409, 605.3021945847591, 800.6462379533782, 715.00686366161, 679.5435060373419, 932.211282013286, 1073.1441247411763, 1203.299007297393, 637.6666788459181, 808.0006875280341, 614.9687752741319, 464.0307557669842, 596.9516874437733, 761.7626956473046, 574.1850688342096, 425.80734004164634, 570.0181609184918, 714.8318813341067, 609.7701387575889, 500.9929066891148]],
    "thumb.jpg": [[0.0, 263.7752018450904, 550.4176488517755, 804.6253824559083, 987.157706362063, 597.239956189808, 787.605393639287, 631.4306891115666, 514.1538875071678, 533.4121180660254, 725.8652583900939, 526.13731381335, 408.04717024476645, 498.5878903448287, 684.0062654717576, 504.25053298384313, 376.2239574767433, 484.2988943679852, 647.8305678611181, 535.5611613081045, 424.56212740767955]],
    "middle.jpg": [[0.0, 317.19881934832387, 603.9877381536918, 807.7835316879529, 780.4230082631834, 589.5018463551831, 779.3651930983651, 663.1812298899889, 549.9668270844378, 568.587275263735, 812.5550036091714, 972.8243013225051, 1116.4812471875766, 538.2491528541009, 708.6899616047831, 545.5213804628639, 397.31654784982373, 528.8973834553675, 703.7057707842816, 611.1881002658549, 487.77021779433585]],
    "ring.jpg": [[0.0, 300.3785481589364, 605.6663633093929, 787.3567339977965, 756.2619963195164, 619.2393181441652, 756.8186246615616, 635.3087423933276, 528.4835988699364, 572.5672713394049, 735.9824759276225, 628.1351054049702, 547.9458558107963, 535.5792994394458, 783.8071690120884, 935.9572592637786, 1070.8929557270467, 504.40823200172355, 622.6695724906187, 527.0433238607274, 416.5372144231996]],
    "pinky.jpg": [[0.0, 297.1275332287322, 594.7009937919036, 790.3669355227928, 791.9543930740207, 601.8254420119489, 785.317556814035, 682.4820030819084, 554.0324888589054, 569.6406797127066, 772.1677437642002, 608.4082825443286, 448.9563493153605, 564.1221205565303, 754.3820779561639, 587.2667861660808, 425.5892970008836, 575.7661493212305, 774.1027063035668, 908.4934883628148, 1027.2037965576744]]
}

# Convert dictionary to NumPy array
keypoints_array = np.array([v[0] for v in image_keypoints_dict.values()])

# Initialize and fit scalers
min_max_scaler = MinMaxScaler().fit(keypoints_array)
standard_scaler = StandardScaler().fit(keypoints_array)
robust_scaler = RobustScaler().fit(keypoints_array)
max_abs_scaler = MaxAbsScaler().fit(keypoints_array)

# Transform the data using each scaler
image_keypoints_dict_MinMax = {k: v for k, v in zip(image_keypoints_dict.keys(), min_max_scaler.transform(keypoints_array))}
image_keypoints_dict_ZScore = {k: v for k, v in zip(image_keypoints_dict.keys(), standard_scaler.transform(keypoints_array))}
image_keypoints_dict_Robust = {k: v for k, v in zip(image_keypoints_dict.keys(), robust_scaler.transform(keypoints_array))}
image_keypoints_dict_MaxAbs = {k: v for k, v in zip(image_keypoints_dict.keys(), max_abs_scaler.transform(keypoints_array))}



# Function to find the closest finger
def find_closest_finger(distances, data):
    min_distance = float('inf')
    closest_finger = None
    for finger, values in data.items():
        distance = euclidean_distance(distances, values[0])
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
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract all 21 landmarks (including wrist)
            landmarks = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
            wrist_pos = landmarks[0]  # Wrist is the first landmark
            distances = [euclidean_distance(wrist_pos, landmark) for landmark in landmarks]

            # Apply scaling
            min_max_scaled_distances = min_max_scaler.transform([distances])[0]
            z_score_distances = standard_scaler.transform([distances])[0]
            robust_scaled_distances = robust_scaler.transform([distances])[0]
            max_abs_scaled_distances = max_abs_scaler.transform([distances])[0]

            # Find the closest finger
            closest_finger_min_max = find_closest_finger(min_max_scaled_distances, image_keypoints_dict_MinMax)
            closest_finger_z_score = find_closest_finger(z_score_distances, image_keypoints_dict_ZScore)
            closest_finger_robust = find_closest_finger(robust_scaled_distances, image_keypoints_dict_Robust)
            closest_finger_max_abs = find_closest_finger(max_abs_scaled_distances, image_keypoints_dict_MaxAbs)

            # Display results
            cv2.putText(frame, f"Min-Max: {closest_finger_min_max}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Z-Score: {closest_finger_z_score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Robust: {closest_finger_robust}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Abs: {closest_finger_max_abs}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()