"""
Event Classification Based on Image Analysis

Summary:
This script loads a pre-trained deep learning model to classify images depicting various war events 
caused by alien attacks into specific categories. The model predicts the event category for each image 
found in the specified folder.

Event Categories:
1. Combat
2. Destroyed Building
3. Fire

4. Humanitarian Aid
5. Military Vehicles

Author: Your Name
Date: June 20, 2024
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define the event names
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

def classify_event(image_path, model):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(180,180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0 
    
    # Predict classes
    predictions = model.predict(img_array)
    class_index = tf.argmax(predictions[0]).numpy()

    if class_index == 0:
        return combat
    elif class_index == 1:
        return destroyed_building
    elif class_index == 2:
        return fire
    elif class_index == 3:
        return rehab
    elif class_index == 4:
        return military_vehicles
    else:
        return "unknown"

def main():
    # Load the pre-trained model
    model = load_model("modelpr.keras")
    
    # Path to the folder containing images
    folder_path = "testing"

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            detected_event = classify_event(image_path, model)
            print(f"Detected Event for {filename}: {detected_event}")

if __name__ == "__main__":
    main()
    
