#unsere main datei
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
#Erstmal den Datensatz herunterladen und in den repository links manuel hinzufügen
'''
url = "https://myshare.leuphana.de/?t=dde5dfe5773fb088bd895a74b49933ab"
tf.keras.utils.get_file("For_model", url, cache_dir="dataset", extract=True)
'''
DIRECTORY = r"Dataset"
CATEGORIES = ["maskchin", "maskmouth", "maskoff", "maskon"]

data = []
labels = []

for category in CATEGORIES:
	path = os.path.join(DIRECTORY, category)
	for img in os.listdir(path):
		img_path = os.path.join(path, img)
		image = load_img(img_path, target_size=(224, 224))
		image = img_to_array(image)
		image = preprocess_input(image)

		data.append(image)
		labels.append(category)
"""
img_height, img_width = 224, 224
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, class_mode='binary', target_size=(img_height, img_width), batch_size=batch_size, shuffle=True, seed=999, subset='training')
val_data = train_datagen.flow_from_directory(train_dir, class_mode='binary', target_size=(img_height, img_width), batch_size=batch_size, shuffle=True, seed=999, subset='validation')
test_data = test_datagen.flow_from_directory(test_dir, class_mode='binary', target_size=(img_height, img_width), batch_size=batch_size,  shuffle=False)
"""