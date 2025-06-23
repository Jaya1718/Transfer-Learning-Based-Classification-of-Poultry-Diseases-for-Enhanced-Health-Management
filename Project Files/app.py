import os
import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("model.h5")

# Class labels
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['pc_image']
        if file:
            # Save the uploaded image
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Preprocess the image
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            return render_template('contact.html', predict=predicted_class)

    # For GET request, just render the page
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
