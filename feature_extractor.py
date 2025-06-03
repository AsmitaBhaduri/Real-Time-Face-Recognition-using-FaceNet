import os
import pickle
from tqdm import tqdm
from deepface import DeepFace
import numpy as np


filenames = pickle.load(open('filenames.pkl', 'rb'))

def feature_extractor(img_path):
    
    embedding = DeepFace.represent(img_path=img_path, model_name='Facenet', enforce_detection=False)

    if embedding and isinstance(embedding, list):
        return np.array(embedding[0]['embedding'])
    else:
        return np.zeros(128)

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file))


pickle.dump(features, open('embedding.pkl', 'wb'))
