import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

os.makedirs('uploads', exist_ok=True)

feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

detector = MTCNN()


def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False


def extract_features(img_path):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if len(results) == 0:
        st.error("No face detected in the uploaded image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]

    face_img = Image.fromarray(face)
    face_img = face_img.resize((160, 160))  #facenet expects 160x160
    face_array = np.asarray(face_img)

    embedding = DeepFace.represent(face_array, model_name='Facenet', enforce_detection=False)
    if embedding and isinstance(embedding, list):
        return np.array(embedding[0]['embedding']) #extracting 128 dim vec from deepface output 
    else:
        return None


def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title('Which Bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        img_path = os.path.join('uploads', uploaded_image.name)
        display_image = Image.open(img_path)

        features = extract_features(img_path)
        if features is not None:
            index_pos = recommend(feature_list, features)
            predicted_actor = filenames[index_pos].split(os.sep)[1].replace('_', ' ')

            col1, col2 = st.columns(2)

            with col1:
                st.header('Your uploaded image')
                st.image(display_image)

            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
