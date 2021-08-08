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
CATEGORIES = ["maskchin", "maskmouth", "maskoff", "maskon"] #wahrscheinlich nicht nötig

img_height, img_width = 224, 224
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
#test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(DIRECTORY, class_mode='categorical', target_size=(img_height, img_width),
                                              batch_size=batch_size, shuffle=True, seed=999, subset='training')
val_data = train_datagen.flow_from_directory(DIRECTORY, class_mode='categorical', target_size=(img_height, img_width),
                                             batch_size=batch_size, shuffle=True, seed=999, subset='validation')
#test_data = test_datagen.flow_from_directory(DIRECTORY, class_mode='categorical', target_size=(img_height, img_width), batch_size=batch_size,  shuffle=False)

#because we have 4 classes? keine Ahnung
num_classes = 4

#adding MobileNetV2 as base model and freezing it before fine tuning
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False)
base_model.trainable = False
model = tf.keras.models.Sequential()

#construction of the top layers
model.add(tf.keras.layers.Input(shape=(img_height, img_width, 3)))
model.add(base_model)
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

model.summary()

#running the model
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10, validation_data=val_data)

#saving the model
model.save("mask_detector_1.model", save_format="h5")
