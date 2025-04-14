import os
import numpy as np
import torch
from flask import Flask, request, jsonify
from torchvision import models, transforms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import json
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv()
import tf_keras as tfk
# import tensorflow_hub as hub

app = Flask(__name__)
CORS(app)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
# acne_model = torch.load('./models/best_model.pth')  # Adjust the path to your .pth file
# acne_model.eval()  # Set model to evaluation mode


disease_model = tfk.models.load_model('./models/skin.h5', custom_objects={'KerasLayer': hub.KerasLayer})
ageing_model = load_model('./models/ageing_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

class_labels = ['dark spots', 'puffy eyes', 'wrinkles']


labels = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of the model
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])


def predict_image(img):
    # Convert the image to RGB if it's not already
    img = img.convert("RGB")

    # Resize the image to the target size
    img = img.resize((224, 224))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image array
    img_array = img_array / 255.0

    # Predict using the model
    predictions_disease = disease_model.predict(img_array)
    predicted_disease_class_idx = np.argmax(predictions_disease)

    # Get the corresponding label
    predicted_disease_class_label = labels[predicted_disease_class_idx]

    predictions_ageing = ageing_model.predict(img_array)

    # Get the predicted label index
    predicted_ageing_class_idx = np.argmax(predictions_ageing, axis=1)

    # Map the index to the corresponding label
    predicted_ageing_label = class_labels[predicted_ageing_class_idx[0]]
    
    return {
        "disease_prediction": predicted_disease_class_label,
        "ageing_prediction": predicted_ageing_label
    }

# def predict_acne_image(img):
#     # Apply transformations to the image
#     img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

#     # Predict using the model
#     with torch.no_grad():
#         outputs = acne_model(img_tensor)
#         _, predicted_class = torch.max(outputs, 1)  # Get the index of the class with the highest score
    
#     # Map the predicted class index to the label
#     return predicted_class.item()  # Return as a simple integer (or string if you have labels)


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



def gemini_suggestions(predicted_label_skin_condition, predicted_label_ageing_condition):
    
    try:
        input_data = f"The detected skin conditions are: {predicted_label_skin_condition} (Skin Disease), {predicted_label_ageing_condition} (Skin Condition)."

        prompt = (
            "Provide a detailed skincare recommendation and lifestyle tips based on the detected skin conditions in JSON format. "
            "The response should include the following structured data: "
            "1. recommendations.coreIngredients: A list of core ingredients to include in skin care that would help treat the current skin condition (e.g., 'Folic Acid', 'Salicylic Acid', 'Retinol'). "
            "2. recommendations.food: A list of food items that promote healthy skin (e.g., 'Avocado', 'Blueberries'). "
            "3. recommendations.suggestedProducts: A list of simple skincare products that can be used for the condition and disease (e.g., 'Gentle Cleanser', 'Moisturizer'). "
            "4. recommendations.lifestyleTips: A list of 5 lifestyle changes or tips to cure the skin disease and reduce effects of condition (e.g., 'Drink plenty of water', 'Get enough sleep'). "
            "Return the data in the following structured format in JSON. Make sure each field has at least 3 values."
            "DON'T GIVE ANY ADDITIONAL INFORMATION OTHER THAN JSON OBJECT."
        )

        # input_data = f"The detected skin condition is: {predicted_label}."
        gemini_response = get_gemini_response(input_data, prompt)
        cleaned_json = clean_json_string(gemini_response)

        # Return the prediction and Gemini response
        return {
            "predicted_label_ageing": predicted_label_ageing_condition,
            "predicted_label_disease": predicted_label_skin_condition,
            "gemini_response": cleaned_json
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the image file and process it
        img = Image.open(io.BytesIO(file.read()))

        # Get the predicted label
        predicted_result  = predict_image(img)  
        predicted_label_skin_condition = predicted_result["disease_prediction"]
        predicted_label_ageing_condition = predicted_result["ageing_prediction"]

        result = gemini_suggestions(predicted_label_skin_condition, predicted_label_ageing_condition)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image part"}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         # Read the image file and process it
#         img = Image.open(io.BytesIO(file.read()))
#         predicted_label = predict_image(img)
#         return jsonify({
#             "predicted_label":  predicted_label,
#             # "gemini_response": clean_json_string(gemini_response)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
    
if __name__ == '__main__':
    app.run(debug=True)
