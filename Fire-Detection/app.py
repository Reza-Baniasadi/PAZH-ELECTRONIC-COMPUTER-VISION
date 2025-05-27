import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

model = load_model('fire_detection_model.h5')

img_size = (224, 224)

def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def app():
    st.title('Fire Detection App')

    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        img = preprocess_image(img)

        pred = model.predict(img)
        pred_label = 'Fire Present' if pred[0][0] > 0.1 else 'No Fire'
        pred_prob = pred[0][0]
        
        st.write(f'Prediction: {pred_label}')
        st.write(f'Probability Of Fire: {pred_prob:.2f}')

if __name__ == '__main__':
    app()