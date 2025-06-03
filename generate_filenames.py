import os
import pickle

filenames = []
actors = os.listdir('data')

for actor in actors:
    actor_folder = os.path.join('data', actor)
    for file in os.listdir(actor_folder):
        file_path = os.path.join(actor_folder, file)
        filenames.append(file_path)

pickle.dump(filenames, open('filenames.pkl', 'wb'))
