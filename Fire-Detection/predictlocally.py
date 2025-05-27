import os
import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('fire_detection_model.h5')

# Set path to the video file
video_file = 'fire.mp4'

# Set the size of the input images
img_size = (224, 224)

# Open the video file
cap = cv2.VideoCapture(video_file)

# Loop through each frame of the video
while True:
    # Read the frame from the video
    ret, frame = cap.read()
    
    # Stop if the video has ended
    if not ret:
        break
    
    # Resize the frame to match the input size of the model
    img = cv2.resize(frame, img_size)
    
    # Normalize the image
    img = img / 255.0
    
    # Make a prediction on the image using the model
    pred = model.predict(np.expand_dims(img, axis=0))[0][0]
    
    # Draw a red box around the area where the fire is detected if the prediction is above a threshold
    if pred > 0.1:
        # Find the coordinates of the top-left and bottom-right corners of the box
        h, w, _ = img.shape
        x1, y1 = int(w/4), int(h/4)
        x2, y2 = int(w*3/4), int(h*3/4)
        
        # Draw the box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        
        # Display the predicted value on the frame
        cv2.putText(frame, f'Fire detected ({pred:.2f})', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
    
    # Display the frame
    cv2.imshow('Fire Detection', frame)
    
    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
