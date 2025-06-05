import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configuration de l'application Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/user-upload/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASSES = [
    "Plante saine", "Bactériose", "Cercosporiose", "Anthracnose", "Mildiou",
    "Rouille", "Mosaïque virale", "Taches foliaires", "Oïdium", "Carence nutritive"
]

MODEL = None

def load_keras_model():
    global MODEL
    try:
        MODEL = load_model('model/plant_disease_model.h5')
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image_path):
    global MODEL
    if MODEL is None:
        load_keras_model()
        if MODEL is None:
            return {"success": False, "error": "Modèle non chargé"}

    start_time = time.time()
    try:
        image = preprocess_image(image_path)
        predictions = MODEL.predict(image)
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_results = [
            {"class": CLASSES[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
            for i in top_indices
        ]
        duration = round(time.time() - start_time, 2)
        return {
            "success": True,
            "prediction": top_results[0]["class"],
            "confidence": top_results[0]["confidence"],
            "top_results": top_results,
            "time_taken": duration,
            "image_path": '/' + image_path.replace("\\", "/")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# === ROUTES ===
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

# ✅ Route GET pour la page HTML
@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')



# ✅ Route POST pour le traitement
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Aucun fichier envoyé"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Aucun fichier sélectionné"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_disease(filepath)
        result["image_path"] = f"/{filepath.replace(os.sep, '/')}"
        return jsonify(result)

    return jsonify({"success": False, "error": "Format de fichier non autorisé"})

if __name__ == '__main__':
    load_keras_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
