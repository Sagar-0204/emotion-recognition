import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load and yield images/labels on-the-fly
def parse_image(pixels, label):
    img = tf.strings.split(pixels)
    img = tf.strings.to_number(img, out_type=tf.float32)
    img = tf.reshape(img, (48, 48, 1))
    img = tf.image.resize(img, (224, 224))
    img = tf.image.grayscale_to_rgb(img)
    img = img / 255.0
    return img, label

def load_dataset(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)
    X = df['pixels'].values
    y = df['emotion'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    train_ds = train_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# Load datasets
train_dataset, val_dataset = load_dataset("C:/Users/tiran/emotion_project/fer2013.csv/fer2013.csv", batch_size=32)

# MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_dataset, validation_data=val_dataset, epochs=50)

# Save model
model.save("emotion_model_mobilenetv2.h5")
