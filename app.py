import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Inisialisasi Flask app
app = Flask(__name__)

# Path untuk model yang telah disimpan
model_path = 'cnn_model.h5'

# Memuat model
model = load_model(model_path)

# Fungsi untuk memproses gambar
def preprocess_image(image):
    img = load_img(image, target_size=(50, 50))  # Ubah ukuran gambar sesuai model
    img_array = img_to_array(img) / 255.0  # Normalisasi gambar
    return np.expand_dims(img_array, axis=0)

# Endpoint untuk menerima upload gambar
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'message': 'Tidak ada file gambar yang ditemukan.'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'message': 'Nama file tidak valid.'}), 400
    
    # Simpan gambar sementara
    image_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(image_path)

    try:
        # Proses gambar
        image_preprocessed = preprocess_image(image_path)
        
        # Prediksi gambar menggunakan model
        prediction = model.predict(image_preprocessed)
        
        # Tentukan hasil prediksi
        predicted_class = 'Mobil' if prediction[0] > 0.5 else 'Bukan Mobil'
        confidence = prediction[0] if prediction[0] > 0.5 else (1 - prediction[0])
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        }), 200
    
    except Exception as e:
        return jsonify({'message': 'Terjadi kesalahan dalam memproses gambar.'}), 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
