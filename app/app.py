
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the 'uploads' directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your pre-trained model
model = tf.keras.models.load_model('C:/Users/salman.nishan/Documents/Personal/Banpu_5/test_8.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img = np.array(img)
    
    # Ensure that the image has three channels (for RGB images)
    if img.shape[-1] != 3:
        img = img[:, :, :3]  # Keep only the first three channels
    
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make predictions using the model
        input_data = preprocess_image(filepath)
        prediction = model.predict(input_data)
        pred = prediction[0][0]
        print(prediction, pred)

        # You can customize this part based on your model's output
        pred_result = f'Model prediction: {pred}'
        print(pred_result)

        label = "GOOD WELD" if pred > 0.8 else "BAD WELD"
        class_label = f"Predicted class: {label}"

        return render_template('index.html', result=pred, class_label=label)

    else:
        return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True, port = 5002)
