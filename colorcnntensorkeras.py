import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow import keras
from keras.utils import to_categorical

img_width = 100
img_height = 100

# dataset in oldugu dizinleri okur
data_dir = os.path.join("C:", os.sep, "Datasets", "VehicleColorsAll")
image_dirs = os.listdir(data_dir)

# sinif sayisi
num_classes = len(image_dirs)

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
            image = cv2.resize(image, (img_width, img_height))
            # listelere ekle
            images.append(image)
            labels.append(folder)

X = np.array(images)
y = np.array(labels)

# piksel degerlerini [0, 1] normalize et
X = X.astype('float32') / 255.0

# veri setini ayir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN modelini olustur
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# etiketleri one-hot kodlamaya donustur
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = to_categorical(le.transform(y_train), num_classes)
le = preprocessing.LabelEncoder()
le.fit(y_test)
y_test = to_categorical(le.transform(y_test), num_classes)

# modeli egit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# test seti ile modeli degerlendir
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# modeli kaydet
model.save('vehiclecolormodel.h5')
