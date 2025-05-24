# AI-Based-copyright-violation-Detector-for-Video-Content.
his project uses *machine learning* and *computer vision* techniques to analyze videos and detect potential copyright violations. It aims to help content creators by providing insights into whether their clips might trigger copyright claims based on visual or audio similarities with existing media.
 

code. 

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to match dimensions
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Calculate Structural Similarity Index (SSI)
    similarity_index, _ = ssim(img1, img2, full=True)

    return similarity_index

# Example Usage
image1 = "sample_clip_frame1.jpg"
image2 = "known_copyrighted_image.jpg"

similarity = compare_images(image1, image2)
print(f"Similarity Score: {similarity}")