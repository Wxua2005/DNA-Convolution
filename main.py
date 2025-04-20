import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time

from dna_encryption import encrypt_image, decrypt_image
from utils import (
    load_image, save_image, plot_images, 
    histogram_analysis, calculate_psnr
)

def main():
    key_params = {
        'x0': 0.6523,  
        'r': 3.9654    
        # 'r' : 3.5713
    }
    rule_number = 3    

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = 'input/phot.jpg'  
    try:
        original_image = load_image(image_path, as_gray=True)
        print(f"Loaded image with shape: {original_image.shape}")

    except Exception as e:
        print(f"Error loading image: {e}")
        print("Creating a sample image...")
        original_image = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                original_image[i, j] = (i + j) % 256

    print("Encrypting image...")
    start_time = time()
    encrypted_image = encrypt_image(original_image, key_params, rule_number)
    encryption_time = time() - start_time
    print(f"Encryption completed in {encryption_time:.2f} seconds")

    print("Decrypting image...")
    start_time = time()
    decrypted_image = decrypt_image(encrypted_image, key_params, rule_number)
    decryption_time = time() - start_time
    print(f"Decryption completed in {decryption_time:.2f} seconds")

    save_image(encrypted_image, os.path.join(output_dir, 'encrypted.png'))
    save_image(decrypted_image, os.path.join(output_dir, 'decrypted.png'))

    plot_images(
        [original_image, encrypted_image, decrypted_image],
        ['Original', 'Encrypted', 'Decrypted']
    )

    histogram_analysis(original_image, encrypted_image)
    
    psnr = calculate_psnr(original_image, decrypted_image)
    print(f"PSNR between original and decrypted: {psnr:.2f} dB")

if __name__ == "__main__":
    main()