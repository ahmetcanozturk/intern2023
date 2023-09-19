import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from extractcolorfeaturesrgb import *
import matplotlib.pyplot as plt 

# dataset in oldugu dizinleri okur
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

plt.figure()
idx = 1
for N in [3,5,7,10,13,15]:
    axis_x = []
    axis_y = []

    for sizemultiplier in [0.2,0.4,0.6,0.8,1.0]:
        # daha az ornek sayisi icin
        size = int(len(images) * sizemultiplier)
        sample_images = images[0:size]
        sample_labels = labels[0:size]

        # listeleri numpy dizilerine cevir
        X = np.array(sample_images)
        y = np.array(sample_labels)

        # veri setini ayir
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ozellikleri olceklendir
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # KNN siniflandirici
        knn_classifier = KNeighborsClassifier(n_neighbors = N)

        # KNN modelini egit
        knn_classifier.fit(X_train, y_train)
        
        # test seti uzerinde tahminlerde bulun
        y_pred = knn_classifier.predict(X_test)

        # modeli degerlendir
        accuracy = accuracy_score(y_test, y_pred)

        axis_x.append(size)
        axis_y.append(accuracy * 100)

    plt.subplot(3,2,idx)
    plt.plot(axis_x, axis_y, linestyle='--', marker='.', color='b')
    plt.xlabel('train size')
    plt.ylabel('accuracy %')
    plt.title(f'# neighbors : {N}')
    idx = idx + 1

plt.subplots_adjust(wspace=1.0, hspace=1.0)
plt.show()
