import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original_image_path, processed_image_path):
    original = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    processed = cv2.imread(processed_image_path, cv2.IMREAD_COLOR)
    # print(original)
    # print(processed)
    if original is None or processed is None:
        raise FileNotFoundError(f"Could not open one of the images: {original_image_path}, {processed_image_path}")
    
    return psnr(original, processed, data_range=255)

def calculate_ssim(original_image_path, processed_image_path):
    original = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    processed = cv2.imread(processed_image_path, cv2.IMREAD_COLOR)
    
    if original is None or processed is None:
        raise FileNotFoundError(f"Could not open one of the images: {original_image_path}, {processed_image_path}")
    
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    return ssim(original_gray, processed_gray, data_range=255)

def get_image_pairs(folder_path):
    original_images = [f for f in os.listdir(folder_path) if "center" in f]
    processed_images = [f for f in os.listdir(folder_path) if "results" in f]
    print(original_images)
    print(processed_images)
    image_pairs = []
    for original in original_images:
        print(original)
        base_name = original.replace("_center.png", "")
        print(base_name)
        corresponding_processed = [p for p in processed_images if base_name in p]
        print(corresponding_processed)
        if corresponding_processed:
            image_pairs.append((os.path.join(folder_path, original), os.path.join(folder_path, corresponding_processed[0])))
    print(image_pairs)
    return image_pairs

# get_image_pairs("test_images")

def calculate_average_psnr(folder_path):
    image_pairs = get_image_pairs(folder_path)
    total_psnr = 0.0
    count = 0
    
    for original_path, processed_path in image_pairs:
        psnr_value = calculate_psnr(original_path, processed_path)
        print(psnr_value)
        total_psnr += psnr_value
        count += 1
    
    return total_psnr / count if count > 0 else float('nan')

def calculate_average_ssim(folder_path):
    image_pairs = get_image_pairs(folder_path)
    total_ssim = 0.0
    count = 0
    
    for original_path, processed_path in image_pairs:
        ssim_value = calculate_ssim(original_path, processed_path)
        print(ssim_value)
        total_ssim += ssim_value
        count += 1
    
    return total_ssim / count if count > 0 else float('nan')

# Example usage
folder_path = "test_images/"  # Update this path to your folder containing the images
average_psnr = calculate_average_psnr(folder_path)
average_ssim = calculate_average_ssim(folder_path)
print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")
