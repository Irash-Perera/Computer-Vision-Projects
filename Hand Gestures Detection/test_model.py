import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./Hand Gestures Detection/model.p', 'rb'))
model = model_dict['model']

#Initialize the camera
cap = cv2.VideoCapture(0)

#Initialize the mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'Peace',1: 'Loser', 2: 'Noice', 3: 'Swag'}

while True:
    data_aux = []
    x_ =[]
    y_ =[]
    
    ret, frame = cap.read()
    
    H_, W_, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Draw the landmarks
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmark,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
        
        # Extract the landmarks
        for hand_landmark in results.multi_hand_landmarks:
            for landmark in range (len(hand_landmark.landmark)):
                data_aux.append(hand_landmark.landmark[landmark].x)
                data_aux.append(hand_landmark.landmark[landmark].y)
                x_.append(hand_landmark.landmark[landmark].x)
                y_.append(hand_landmark.landmark[landmark].y)
        
        if (len(data_aux) == 42):
            x1 = int(min(x_) * W_) -10
            x2 = int(max(x_) * W_) -10
            
            y1 = int(min(y_) * H_) -10
            y2 = int(max(y_) * H_) -10
            
            prediction = model.predict([np.asarray(data_aux)])
            prediction_label = labels_dict[int(prediction[0])]
            
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),4)
            cv2.putText(frame, prediction_label,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    
        else:
            prediction_label = 'Unpected feature shape'
        print(prediction_label)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF ==27:
        break