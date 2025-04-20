import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image(path, as_gray=True):
    if as_gray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)

def save_image(image, path):
    cv2.imwrite(path, image)

def plot_images(images, titles, figsize=(12, 4)):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        if len(images) == 1:
            ax = axes
        else:
            ax = axes[i]
            
        if len(image.shape) == 3 and image.shape[2] == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_psnr(original, processed):    
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def histogram_analysis(original, encrypted):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(original.flatten(), bins=256, range=[0, 256], color='b', alpha=0.7)
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(encrypted.flatten(), bins=256, range=[0, 256], color='r', alpha=0.7)
    plt.title('Histogram of Encrypted Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()