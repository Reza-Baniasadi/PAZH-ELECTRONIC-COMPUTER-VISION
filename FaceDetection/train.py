import tensorflow as tf
import numpy as np
import os
import cv2 as cv
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    return img_preprocessed

def get_embedding(image_path):
    img = preprocess_image(image_path)
    embedding = base_model(img)
    return embedding.numpy()

dataset_path = '/Users/mac/Downloads/content/Pazh-Electronic-computer-vision/DataSet/pictures'
embeddings = []
names = []

for file in os.listdir(dataset_path):
    if file.lower().endswith('.jpg'):
        path = os.path.join(dataset_path, file)
        emb = get_embedding(path)
        embeddings.append(emb[0])
        name = file.split('_')[0]
        names.append(name)

embeddings = np.array(embeddings)

def recognize_face(face_img, threshold=0.7):
    img_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img_rgb, (224, 224))
    img_normalized = tf.keras.applications.resnet50.preprocess_input(img_resized.astype(np.float32))
    img_expanded = np.expand_dims(img_normalized, axis=0)

    emb = base_model(img_expanded).numpy()

    if len(embeddings) == 0:
        raise ValueError("هیچ دیتاستی برای مقایسه وجود نداد")

    sims = cosine_similarity(emb, embeddings)[0]
    max_sim_idx = np.argmax(sims)
    max_sim = sims[max_sim_idx]

    if max_sim > threshold:
        print(f"فرد شناسایی شد: {names[max_sim_idx]} با شباهت {max_sim}")
        return names[max_sim_idx]
    else:
        raise ValueError("چهره ناشناس است!")
