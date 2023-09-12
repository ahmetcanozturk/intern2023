import cv2
import numpy as np

def extract_features(image):
    # Convert the image to the HSV color space (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range of values for each color channel (Hue)
    hue_ranges = [0, 30, 60, 90, 120, 150, 180]
    
    # Initialize a list to store the histograms for each color channel
    histograms = []
    
    # Calculate the histogram for each color channel
    for i in range(len(hue_ranges) - 1):
        # Create a mask for the current hue range
        lower_hue = hue_ranges[i]
        upper_hue = hue_ranges[i + 1]
        mask = cv2.inRange(hsv_image, (lower_hue, 50, 50), (upper_hue, 255, 255))
        
        # Calculate the histogram for the masked region
        histogram = cv2.calcHist([hsv_image], [0], mask, [256], [0, 256])
       
        # Normalize the histogram
        cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
        
        # Flatten the histogram to a 1D array and append it to the list
        histograms.extend(histogram.ravel())
    
    # Convert the list of histograms to a NumPy array
    feature_vector = np.array(histograms)
    
    return feature_vector