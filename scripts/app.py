import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('./Food.keras')

# Define class names
CLASS_NAMES = ['Besan_cheela', 'Biryani', 'Chapathi', 'Chole_bature', 'Dahl', 'Dhokla', 'Dosa', 'Gulab_jamun',
               'Idli', 'Jalebi', 'Kadai_paneer', 'Naan', 'Paani_puri', 'Pakoda', 'Pav_bhaji', 'Poha', 'Rolls', 'Samosa',
               'Vada_pav', 'chicken_curry', 'chicken_wings', 'donuts', 'fried_rice', 'grilled_salmon', 'hamburger',
               'ice_cream', 'not_food', 'pizza', 'ramen', 'steak', 'sushi']

@app.route('/', methods=['GET'])
def home():
    """Home route to check if API is running"""
    return "Keras Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data found in request'}), 400
        
        # Get base64 image data
        base64_image = data['image_data']
        
        # If the base64 string contains a header (e.g., "data:image/jpeg;base64,"), remove it
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(base64_image)
        
        # Convert bytes to tensor
        img = tf.image.decode_image(img_bytes, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.expand_dims(img, axis=0)
        
        # Make prediction
        pred = model.predict(img)
        
        predicted_class_index = np.argmax(pred[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence_score = float(np.max(pred[0]))
        
        return jsonify({
            'prediction': predicted_class, 
            'confidence': confidence_score
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)