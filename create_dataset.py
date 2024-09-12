import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Hand Gestures Detection/data'

# Create two arrays to store the landmarks data and labels
data = []
labels = []

# Iterate through all images and extract the landmarks

for dir_ in os.listdir(DATA_DIR):
    for img in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = [] # Store the landmarks for each image
        image = cv2.imread(os.path.join(DATA_DIR, dir_, img))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                for landmark in range (len(hand_landmark.landmark)):
                    data_aux.append(hand_landmark.landmark[landmark].x)
                    data_aux.append(hand_landmark.landmark[landmark].y)  
                # mp_drawing.draw_landmarks(image_rgb,
                #                           hand_landmark,
                #                           mp_hands.HAND_CONNECTIONS,
                #                           mp_drawing_styles.get_default_hand_landmarks_style(),
                #                           mp_drawing_styles.get_default_hand_connections_style())
        # plt.figure()
        # plt.imshow(image_rgb)
# plt.show()

        data.append(data_aux)
        labels.append(dir_)

f = open('./Hand Gestures Detection/data.pkl', 'wb')
pickle.dump({'data': data, 'labels': labels},f)
f.close()
            
            
        
        
                    
            
        
        
