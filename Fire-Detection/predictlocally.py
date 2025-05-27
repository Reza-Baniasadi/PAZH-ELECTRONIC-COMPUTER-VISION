import os
import numpy as np
import cv2
from keras.models import load_model

model = load_model('fire_detection_model.h5')

video_file = 'fire.mp4'

img_size = (224, 224)

cap = cv2.VideoCapture(video_file)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    img = cv2.resize(frame, img_size)
    
    img = img / 255.0
    
    pred = model.predict(np.expand_dims(img, axis=0))[0][0]
    
    if pred > 0.1:
        h, w, _ = img.shape
        x1, y1 = int(w/4), int(h/4)
        x2, y2 = int(w*3/4), int(h*3/4)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        
        cv2.putText(frame, f'Fire detected ({pred:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
    
    cv2.imshow('Fire Detection', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
