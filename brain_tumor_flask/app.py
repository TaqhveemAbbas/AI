from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Add this new constant at the top of your file
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.static_folder = STATIC_FOLDER

# Load the trained model
model = load_model(r"C:\Users\user\Downloads\archive (31)\brain_tumor_flask\brain_tumor_detection_model.keras")

categories = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/view3d')
def view3d():
    return render_template('viewer.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            # Save the file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Preprocess the image
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_category = categories[predicted_index]
            probability = float(np.max(predictions) * 100)
            
            return jsonify({
                'predicted_class': predicted_category,
                'probability': round(probability, 2)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        print(f"Attempting to serve: {filename}")
        print(f"Full path: {os.path.join(STATIC_FOLDER, filename)}")
        # Split the path to get the directory and file
        directory = os.path.dirname(filename)
        file = os.path.basename(filename)
        # Serve from the appropriate subdirectory
        return send_from_directory(os.path.join(STATIC_FOLDER, directory), file)
    except Exception as e:
        print(f"Error serving {filename}: {str(e)}")
        return str(e), 404

@app.route('/check_file')
def check_file():
    model_path = os.path.join(STATIC_FOLDER, 'models', 'brain.glb')
    exists = os.path.exists(model_path)
    return {
        'file_exists': exists,
        'file_path': model_path,
        'static_folder': STATIC_FOLDER
    }

if __name__ == '__main__':
    app.run(debug=True)
