import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import multiprocessing

from dna_encryption import encrypt_image, decrypt_image
from utils import (
    load_image, save_image, plot_images, 
    histogram_analysis, calculate_psnr
)

def main():
    key_params = {
        'x0': 0.6523,  
        'r': 3.9654    
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

    num_threads = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_threads} threads for processing")

    print("\nEncrypting image with multithreading...")
    start_time = time()
    encrypted_image = encrypt_image(original_image, key_params, rule_number, num_threads=num_threads)
    encryption_time = time() - start_time
    print(f"Multithreaded encryption completed in {encryption_time:.2f} seconds")

    print("\nDecrypting image with multithreading...")
    start_time = time()
    decrypted_image = decrypt_image(encrypted_image, key_params, rule_number, num_threads=num_threads)
    decryption_time = time() - start_time
    print(f"Multithreaded decryption completed in {decryption_time:.2f} seconds")

    save_image(encrypted_image, os.path.join(output_dir, 'encrypted_threaded.png'))
    save_image(decrypted_image, os.path.join(output_dir, 'decrypted_threaded.png'))

    plot_images(
        [original_image, encrypted_image, decrypted_image],
        ['Original', 'Encrypted (Threaded)', 'Decrypted (Threaded)']
    )

    histogram_analysis(original_image, encrypted_image)
    
    psnr = calculate_psnr(original_image, decrypted_image)
    print(f"PSNR between original and decrypted: {psnr:.2f} dB")

    if False:  
        print("\nRunning single-threaded version for comparison...")
        start_time = time()
        encrypted_single = encrypt_image(original_image, key_params, rule_number, num_threads=1)
        single_encryption_time = time() - start_time
        print(f"Single-threaded encryption completed in {single_encryption_time:.2f} seconds")
        
        speedup = single_encryption_time / encryption_time
        print(f"Multithreading speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
