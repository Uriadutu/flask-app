import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from io import BytesIO

# Inisialisasi Flask app
app = Flask(__name__)

# Path untuk model yang telah disimpan
model_path = 'cnn_model.h5'

# Memuat model dengan error handling
try:
    model = load_model(model_path)
    print("Model berhasil dimuat.")
except Exception as e:
    model = None
    print(f"Error saat memuat model: {e}")

# Fungsi untuk memproses gambar
def preprocess_image(file):
    try:
        img = Image.open(file).resize((50, 50))  # Resize sesuai dengan input model
        img_array = img_to_array(img) / 255.0  # Normalisasi gambar
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error saat memproses gambar: {e}")

# Endpoint untuk menerima upload gambar
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if model is None:
        return jsonify({'message': 'Model tidak tersedia. Pastikan model telah dimuat dengan benar.'}), 500

    if 'image' not in request.files:
        return jsonify({'message': 'Tidak ada file gambar yang ditemukan.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'message': 'Nama file tidak valid.'}), 400

    try:
        # Proses gambar langsung dari memori
        image_preprocessed = preprocess_image(BytesIO(file.read()))
        
        # Prediksi gambar menggunakan model
        prediction = model.predict(image_preprocessed)
        
        # Tentukan hasil prediksi
        predicted_class = 'Mobil' if prediction[0] > 0.5 else 'Bukan Mobil'
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else (1 - prediction[0][0])
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(float(confidence), 4)  # Batasi confidence ke 4 desimal
        }), 200

    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except Exception as e:
        return jsonify({'message': f'Terjadi kesalahan dalam memproses gambar: {str(e)}'}), 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
