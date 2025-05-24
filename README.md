# Real-Time Emotion Recognition System

This project implements a **real-time facial emotion recognition system** using a **pretrained MobileNetV2 CNN** and OpenCV. It processes webcam video input, detects faces using Haar Cascades, and classifies facial expressions into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## Project Structure

- `emotion_model.py` – Trains a MobileNetV2-based CNN on the FER-2013 dataset and saves the model.
- `webcam_model.py` – Loads the trained model and performs real-time emotion recognition from webcam input.

---

## Features

- ✅ Deep learning model using **MobileNetV2**
- ✅ Efficient training on **FER-2013 facial expression dataset**
- ✅ Real-time emotion classification from **webcam feed**
- ✅ Uses **OpenCV** for face detection and display
- ✅ Accurate and lightweight for deployment

---

## Emotions Detected

- Angry 
- Disgust  
- Fear 
- Happy  
- Sad 
- Surprise  
- Neutral  

---

## Dataset

- **FER-2013**: [Kaggle - Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

---

## Contributing

Pull requests and feedback are welcome! For major changes, please open an issue first to discuss your idea.


## Requirements

Install required libraries using:

```bash
pip install tensorflow opencv-python pandas numpy scikit-learn

