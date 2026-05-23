import numpy as np
import tensorflow as tf
import cv2
import time

def load_model_and_labels():
    model = tf.saved_model.load("converted_savedmodel/model.savedmodel")
    
    labels = {}
    with open("converted_savedmodel/labels.txt", 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                index = int(parts[0])
                label = ' '.join(parts[1:])
                labels[index] = label
    
    return model, labels

def preprocess_image(image, target_size=(224, 224)):
    resized = cv2.resize(image, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched

def predict_class(image, model, labels):
    processed_image = preprocess_image(image)
    predictions = model(processed_image)
    
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]
    label = labels.get(predicted_class, f"Unknown_{predicted_class}")
    
    return {
        'class': predicted_class,
        'label': label,
        'confidence': confidence
    }

model, labels = load_model_and_labels()

def capture_and_classify():
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    result = predict_class(frame, model, labels)['class']
    return result

if __name__ == "__main__":
    while True:
        print(capture_and_classify())