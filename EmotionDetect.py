import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tf_keras
img=cv2.imread('surprise.WEBP')
plt.imshow(img)
#for printing the coloured image
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show()
predictions=DeepFace.analyze(img)

#convert the predictions from list into dict
predictions_dict = predictions[0]

#print(predictions)

#for making rectangle on object's face
faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h)in faces:
    cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0),2)

#for showing rectangular box image

#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show()

#for showing emotion text with rect box
font = cv2.FONT_HERSHEY_SIMPLEX
#here you have to remember that you have changed the type of predictions
cv2.putText(img,predictions_dict['dominant_emotion'],
            (0,50),
            font,1,(0,0,255),2,cv2.LINE_4)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()