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
import base64
import json
import google.generativeai as genai

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
def load_ageing_model():
    try:
        ageing_model = load_keras_model('./models/ageing_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        print("Ageing model loaded successfully.")
        return ageing_model
    except Exception as e:
        print(f"Error loading ageing model: {str(e)}")
        return None

# Preprocess image for both models
def preprocess_image(image_bytes):
    """Preprocess an image for model inference"""
    # Define the same transformation pipeline used during testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Open and transform the image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = test_transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# Predict skin acne severity
def predict_acne(image_tensor):
    # Ensure the acne model is loaded
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

# Predict skin ageing condition
def predict_ageing(img):
    img = img.convert("RGB")  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to 224x224 for MobileNetV2
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = ageing_model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)
    
    ageing_labels = ['dark spots', 'puffy eyes', 'wrinkles']
    return ageing_labels[predicted_class_idx[0]]


def clean_json_string(response):
    # Remove the backticks and 'json' tag
    print(response)
    cleaned_response = response.replace("```json", "").replace("```", "")

    cleaned_response = cleaned_response.strip()

    try:
        json_data = json.loads(cleaned_response)
        print(json_data)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None
    
    return json_data

def get_gemini_response(input_data, prompt):
    try:

        gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        response = gemini_model.generate_content([input_data, prompt])
        return response.text.strip()
    except Exception as e:
        return f"Error during Gemini response generation: {str(e)}"


def gemini_suggestions(acne_result, ageing_result, skin_analysis_result):
    
    try:
        input_data = f"The detected skin conditions are: {acne_result} (Skin Disease), {ageing_result} (Skin Condition). The patient has filled the given survey form: {skin_analysis_result}."

        prompt = (
            "You are a renowned Dermatologist in India. Based on the skin survey form and disease conditions predicted along with skin symptom of ageing, provide a detailed skincare recommendation and lifestyle tips in JSON format. "
            "The response should include the following structured data: "
            "1. recommendations.coreIngredients: A list of core ingredients to include in skin care that would help treat the current skin condition (e.g., 'Folic Acid', 'Salicylic Acid', 'Retinol'). "
            "2. recommendations.food: A list of food items that promote healthy skin (e.g., 'Avocado', 'Blueberries'). "
            "3. recommendations.suggestedProducts: A list of simple skincare products that can be used for the condition and disease (e.g., 'Gentle Cleanser', 'Moisturizer'). "
            "4. recommendations.lifestyleTips: A list of 5 lifestyle changes or tips to cure the skin disease and reduce effects of condition (e.g., 'Drink plenty of water', 'Get enough sleep'). "
            "5. summary: A comprehensive summary analyzing the survey form data and predicted results to help patient understand his overall mellowed down condition and skin type to take care of. DO NOT REPEAT THE RECOOMENDATIONS. Write something different like introducing the patient to his skin type. Be to the point. No introductory line."
            "Return the data in the following structured format in JSON. Make sure each field has at least 3 values."
            "DON'T GIVE ANY ADDITIONAL INFORMATION OTHER THAN JSON OBJECT."
        )

        # input_data = f"The detected skin condition is: {predicted_label}."
        gemini_response = get_gemini_response(input_data, prompt)
        cleaned_json = clean_json_string(gemini_response)

        # Return the prediction and Gemini response
        return {
            "predicted_label_ageing": ageing_result,
            "predicted_label_disease": acne_result,
            "gemini_response": cleaned_json
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-face', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        skin_analysis_result = request.form.get('skin_analysis_result')
     
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        try:
            if file:
                image_bytes = file.read()
                image_tensor = preprocess_image(image_bytes)
                
                # Make acne severity prediction
                acne_result = predict_acne(image_tensor)
                
                # Process the image for the ageing model
                img = Image.open(io.BytesIO(image_bytes))
                ageing_result = predict_ageing(img)
                result = gemini_suggestions(acne_result, ageing_result, skin_analysis_result)

                # return jsonify(result)
                # Return both results
                return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Load models at startup
    load_acne_model()
    ageing_model = load_ageing_model()  # Keras model for ageing prediction
    app.run(debug=True, port=5000)
