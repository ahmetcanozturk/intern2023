import os
import cv2
import numpy as np
from keras.models import load_model

# modeli getir
model = load_model('vehicle_color_model.h5')

# Load and preprocess a new image for prediction
data_dir = os.path.join("C:", os.sep, "Datasets", "VehicleColors", "val")
new_image_folder = "orange"
new_image_name = "8de6a236d7.jpg"
new_image_path = os.path.join(data_dir, new_image_folder, new_image_name)
new_image = cv2.imread(new_image_path)
# imaji olceklendir
new_image = cv2.resize(new_image, (100, 100))
# piksel degerlerini normalize et
new_image = new_image.astype('float32') / 255.0

# yeni resim hakkında tahminde bulunun
predictions = model.predict(np.expand_dims(new_image, axis=0))

# tahmini yorumla
data_dir = os.path.join("C:", os.sep, "Datasets", "VehicleColorsAll")
class_labels = os.listdir(data_dir)
predicted_class = np.argmax(predictions)
predicted_color = class_labels[predicted_class]

print(f"Predicted color: {predicted_color}")