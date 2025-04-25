from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import tf_keras as tfk
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model as load_keras_model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define your model class for Acne Classification (PyTorch model)
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.cnn = torchvision.models.efficientnet_v2_m(weights=None)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.classifier = nn.Sequential(
            nn.Linear(self.cnn.classifier[1].in_features, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Linear(64, 4),  # 4 severity classes for acne
        )
        
    def forward(self, img):
        output = self.cnn(img)
        return output

# Load the trained PyTorch model for acne classification
acne_model = None

def load_acne_model():
    global acne_model
    acne_model = MyNet()
    acne_model.load_state_dict(torch.load('./models/acne_classification_model.pth', map_location=torch.device('cpu')))
    acne_model.eval()
    return acne_model

# Load the trained Keras model for skin ageing classification
ageing_model = None

def load_ageing_model():
    global ageing_model
    try:
        ageing_model = load_keras_model('./models/ageing_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        print("Ageing model loaded successfully.")
        return ageing_model
    except Exception as e:
        print(f"Error loading ageing model: {str(e)}")
        return None

# Load the disease model (TensorFlow Keras + TF Hub)
disease_model_instance = None

def load_disease_model():
    global disease_model_instance
    try:
        # Ensures hub.KerasLayer is available during deserialization
        disease_model_instance = tfk.models.load_model(
            './models/skin.h5',
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        print("Disease model loaded successfully.")
    except Exception as e:
        print(f"Error loading disease model: {str(e)}")
        disease_model_instance = None

# Preprocess image for PyTorch model
def preprocess_image(image_bytes):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# Predict acne severity (PyTorch)
def predict_acne(image_tensor):
    global acne_model
    if acne_model is None:
        acne_model = load_acne_model()

    with torch.no_grad():
        outputs = acne_model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        
    severity_labels = ["Mild", "Moderate", "Severe", "Very Severe"]
    return {
        'class_index': class_idx,
        'severity': severity_labels[class_idx]
    }

# Predict ageing condition (Keras + TF Hub)
def predict_ageing(img):
    global ageing_model
    if ageing_model is None:
        load_ageing_model()

    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = ageing_model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)
    
    ageing_labels = ['dark spots', 'puffy eyes', 'wrinkles']
    return ageing_labels[predicted_class_idx[0]]

# Predict skin disease (Keras + TF Hub)
def predict_disease(img):
    global disease_model_instance
    if disease_model_instance is None:
        load_disease_model()

    if disease_model_instance is None:
        return "Model not available"

    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = disease_model_instance.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)
    
    disease_labels = [
        "BA-cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus",
        "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"
    ]
    return disease_labels[predicted_class_idx[0]]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)

        acne_result = predict_acne(image_tensor)

        img = Image.open(io.BytesIO(image_bytes))
        ageing_result = predict_ageing(img)
        disease_result = predict_disease(img)

        return jsonify({
            'status': 'success',
            'acne_prediction': acne_result,
            'ageing_prediction': ageing_result,
            'disease_prediction': disease_result
        })

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    load_acne_model()
    load_ageing_model()
    load_disease_model()
    app.run(debug=True, port=5000)
