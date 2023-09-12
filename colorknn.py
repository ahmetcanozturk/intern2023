import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from extractcolorfeaturesrgb import *

# Load the dataset (assuming you have a dataset of images and corresponding labels)
data_dir = os.path.join("C:", os.sep, "Datasets", "VehicleColorsAll")
image_dirs = os.listdir(data_dir)

images = []
labels = []

# tum dizinler icerisinde gez
for folder in image_dirs:
    color_dir = os.path.join(data_dir, folder)
    # dizindeki tum dosyalari al
    image_files = os.listdir(color_dir)
    for file in image_files:
        # jpg imaj dosyalarini oku
        if file.endswith(".jpg"):
            image = cv2.imread(os.path.join(color_dir, file))
            # imajlari fix bir boyuta getir
            # image = cv2.resize(image, (100, 100))
            # özellikleri cikar (renk histogrami)
            features = extract_features(image)
            # listelere ekle
            images.append(features)
            labels.append(folder)

# listeleri numpy dizilerine cevir
X = np.array(images)
y = np.array(labels)

# veri setini ayir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ozellikleri olceklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN siniflandirici
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# KNN modelini egit
knn_classifier.fit(X_train, y_train)

# test seti uzerinde tahminlerde bulun
y_pred = knn_classifier.predict(X_test)

# dogrulugu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")