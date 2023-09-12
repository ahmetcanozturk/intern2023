import cv2
import numpy as np

def extract_features(image):
    # imaji rgb kanallarina ayir
    b, g, r = cv2.split(image)

    # her kanal icin bolme sayisi
    num_bins = 256

    # her renk kanali icin histogram hesapla
    hist_r = cv2.calcHist([r], [0], None, [num_bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [num_bins], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [num_bins], [0, 256])

    # histogrami normalize et
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    # histogramlari tek bir ozellik vektorunde birlestir
    feature_vector = np.concatenate((hist_r, hist_g, hist_b))

    return feature_vector