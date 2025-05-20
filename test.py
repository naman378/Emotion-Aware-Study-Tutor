import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tf_keras

# Load image
img = cv2.imread("sadboy.jpg")

# Analyze emotions
predictions = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

# Convert list to dictionary (extract first face's data)
predictions_dict = predictions[0]

# Get face region
face_region = predictions_dict['region']
x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']

# Draw rectangle around the face
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Overlay the detected emotion
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, predictions_dict['dominant_emotion'], (x, y - 10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.axis("off")
plt.show()
print(type(predictions))