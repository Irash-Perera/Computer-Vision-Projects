import cv2
import mediapipe as mp
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./Hand Gestures Detection/data.pkl', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy:', score*100, '%')


