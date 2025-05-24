import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('emotion_model_mobilenetv2.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face from frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        face = cv2.resize(face, (224, 224))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        prediction = model.predict(face)
        emotion_idx = np.argmax(prediction)
        emotion_text = emotion_labels[emotion_idx]

        # Put emotion text above rectangle
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
