from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import glob
import json
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pandas as pd

# --- Config ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load YOLO model ---
model = YOLO("best.pt")  # Make sure 'best.pt' is in the same folder or correct path

# --- Load Food Data ---
food_df = pd.read_csv('food_cleaned.csv')  # Ensure this CSV is present

# --- Helper: Detect food using YOLO ---
def detect_food(image_path):
    results = model(image_path)
    detected = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            detected.add(label)
    return list(detected)

# --- Helper: Get nutrition from CSV ---
def get_nutrition(food_name):
    match = food_df[food_df['Food_Name'].str.lower() == food_name.lower()]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

# --- Helper: Clear uploads folder ---
def clear_upload_folder():
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting file {f}: {e}")

# --- Helper: Validate file type ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper: Save nutrition entry to JSON file ---
def update_nutrition_data(new_entry, file_path='nutrition_data.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):
                    print("Error: JSON data is not a list.")
                    return
            except json.JSONDecodeError:
                print("Error: JSON file is empty or malformed.")
                return
    else:
        data = []

    data.append(new_entry)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"New entry for {new_entry.get('Food_Name', 'Unknown')} added successfully.")

# --- Route: Upload image and detect food ---
@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400

    clear_upload_folder()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    detected_foods = detect_food(file_path)
    nutrients = []

    for food in detected_foods:
        info = get_nutrition(food)
        if info:
            nutrient_info = {
                'Food_Name': info.get('Food_Name'),
                'Calories_per_100g': info.get('Calories_per_100g'),
                'Carbs_g': info.get('Carbs_g'),
                'Fat_g': info.get('Fat_g'),
                'Protein_g': info.get('Protein_g')
            }
            nutrients.append(nutrient_info)
            update_nutrition_data(nutrient_info)

    return jsonify({'nutrients': nutrients}), 200

# --- Start the app ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
