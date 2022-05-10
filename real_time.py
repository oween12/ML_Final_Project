import cv2
import torch
from model import *

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = torch.load("best_model")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face_gray = gray[y:y+h,x:x+w]
        face_gray = cv2.resize(face_gray, (48,48), interpolation=cv2.INTER_AREA)

        if np.sum([face_gray]) != 0:
            flat = np.empty((1,(48 * 48)))
            flat[0] = face_gray.flatten()
            flat_torch = torch.from_numpy(flat.astype(np.float32))
            logits = model(flat_torch)
            label = torch.max(logits, 1)[1]
            emotion = emotion_labels[label]
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

