import cv2
import numpy as np

def extract_features(image):
    # imaji HSV renk alanina cevir
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # her renk kanalı icin deger araligi
    hue_ranges = [0, 30, 60, 90, 120, 150, 180]
    
    histograms = []
    
    # her renk kanali icin histogram hesapla
    for i in range(len(hue_ranges) - 1):
        # renk tonu araligi icin maske olusturma
        lower_hue = hue_ranges[i]
        upper_hue = hue_ranges[i + 1]
        mask = cv2.inRange(hsv_image, (lower_hue, 50, 50), (upper_hue, 255, 255))
        
        # maskelenmis bolge icin histogram olustur
        histogram = cv2.calcHist([hsv_image], [0], mask, [256], [0, 256])
       
        # histogrami normalize et
        cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
        
        # Flatten the histogram to a 1D array and append it to the list
        # histogrami 1B diziye duzlestir ve listeye ekle
        histograms.extend(histogram.ravel())
    
    # listeyi numpy dizisine cevir
    feature_vector = np.array(histograms)
    
    return feature_vector