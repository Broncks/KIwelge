# unsere main datei
import tensorflow as tf
import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Erstmal den Datensatz herunterladen und in den repository links manuel hinzufügen
'''
url = "https://myshare.leuphana.de/?t=dde5dfe5773fb088bd895a74b49933ab"
tf.keras.utils.get_file("For_model", url, cache_dir="dataset", extract=True)
'''
DIRECTORY = r"Dataset"
CATEGORIES = ["maskchin", "maskmouth", "maskoff", "maskon"]

img_height, img_width = 224, 224
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(DIRECTORY, class_mode='categorical', target_size=(img_height, img_width),
                                               batch_size=batch_size, shuffle=True, seed=999, subset='training')
val_data = train_datagen.flow_from_directory(DIRECTORY, class_mode='categorical', target_size=(img_height, img_width),
                                             batch_size=batch_size, shuffle=True, seed=999, subset='validation')


# training visual
"""def plot_training(history):
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Loss')
    plt.xlabel('No. of Epochs')
    plt.ylabel('loss value')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('No. of Epochs')
    plt.ylabel('accuracy value')
    plt.legend()
    plt.show() """


num_classes = 4

base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False)
base_model.trainable = False
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(img_height, img_width, 3)))
model.add(base_model)
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2()))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

model.summary()
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10, validation_data=val_data)


model.save("mask_detector_1.model", save_format="h5")

#dhgsjdgh