import pickle
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
from tqdm import tqdm

feature_list = np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

detector = MTCNN()

# Read test image and detect face
sample_img = cv2.imread('sample/satya.jpg')
results = detector.detect_faces(sample_img)

if len(results) == 0:
    raise Exception("No face detected in sample image.")

x, y, width, height = results[0]['box']
face = sample_img[y:y + height, x:x + width]

# Save or convert face to temporary file to feed to DeepFace (DeepFace expects file path or numpy array)
face_img = Image.fromarray(face)
face_img = face_img.resize((224, 224))
face_array = np.asarray(face_img)

# DeepFace expects either file path or numpy array; passing numpy array here:
embedding = DeepFace.represent(face_array, model_name='Facenet', enforce_detection=False)

if embedding and isinstance(embedding, list):
    result = np.array(embedding[0]['embedding'])
else:
    result = np.zeros(128)

# Compute similarity with all stored embeddings
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]

# Show matched image
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
