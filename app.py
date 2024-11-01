from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the .keras model file
model = load_model('best_brain_tumor_detection_model.keras')  # Replace with the path to your .keras model file
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']  # Adjust according to your model's classes

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle image uploads and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image and make a prediction
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]

    # Delete the uploaded file after processing
    os.remove(file_path)

    return jsonify({"prediction": predicted_class})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
