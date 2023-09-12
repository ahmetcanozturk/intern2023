import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from extractcolorfeaturesrgb import *

# Load the dataset (assuming you have a dataset of images and corresponding labels)
data_dir = os.path.join("C:", os.sep, "CanFiles", "Datasets", "VehicleColorsAll")
image_dirs = os.listdir(data_dir)

images = []
labels = []

for folder in image_dirs:
    color_dir = os.path.join(data_dir, folder)
    image_files = os.listdir(color_dir)
    for file in image_files:
        if file.endswith(".jpg"):
            image = cv2.imread(os.path.join(color_dir, file))
            # Resize the image to a fixed size (e.g., 100x100 pixels)
            # image = cv2.resize(image, (100, 100))
            # Extract relevant features (e.g., color histograms)
            # You may need to experiment with different feature extraction methods
            features = extract_features(image)
            images.append(features)
            labels.append(folder)

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the KNN model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")