import os
import cv2
import numpy as np

data_dir = os.path.join("C:", os.sep, "CanFiles", "Datasets", "VehicleColorsAll", "red")
image_file = os.path.join(data_dir, "0d98f6d8b4.jpg")
# Load the image containing the vehicle
image = cv2.imread(image_file)

# Convert the image to the HSV color space (Hue, Saturation, Value)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the color range for the vehicle
# You should adjust these values based on the color of the vehicle in your images
h_min = 0
s_min = 80
v_min = 75
h_max = 0
s_max = 100
v_max = 100
lower_color = np.array([h_min, s_min, v_min])  # Lower bound of color range
upper_color = np.array([h_max, s_max, v_max])  # Upper bound of color range

# Create a mask that isolates the vehicle based on color range
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Apply bitwise AND operation to the original image and the mask to extract the vehicle
extracted_vehicle = cv2.bitwise_and(image, image, mask=mask)

# Save or display the extracted vehicle
cv2.imshow('Vehicle', image)
cv2.imshow('Extracted Vehicle', extracted_vehicle)
cv2.waitKey(0)
cv2.destroyAllWindows()