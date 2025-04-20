import numpy as np
import cv2
import threading
import concurrent.futures
from functools import partial

from dna_operations import (
    encode_image_to_dna, decode_dna_to_image,
    dna_xor_operation
)
from chaotic_maps import generate_key

DNA_KERNELS = {
    'A': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  
    'C': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  
    'G': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),   
    'T': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  
}

def _process_convolution_chunk(chunk_data):
    start_row, end_row, dna_image, kernel, height, width = chunk_data
    result = np.empty((end_row - start_row, width), dtype='object')
    
    for i_rel, i in enumerate(range(start_row, end_row)):
        for j in range(width):
            i_start, i_end = max(0, i-1), min(height, i+2)
            j_start, j_end = max(0, j-1), min(width, j+2)
            
            neighborhood = np.zeros((3, 3, 4), dtype='object')
            
            for ni, i_idx in enumerate(range(i_start, i_end)):
                for nj, j_idx in enumerate(range(j_start, j_end)):
                    if 0 <= i_idx < height and 0 <= j_idx < width:
                        dna_seq = dna_image[i_idx, j_idx]
                        for k in range(min(4, len(dna_seq))):
                            neighborhood[ni, nj, k] = dna_seq[k]
            
            dna_result = ''
            for k in range(4):
                nucleotides = [neighborhood[ni, nj, k] for ni in range(3) for nj in range(3)]
                counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
                for nucleotide, weight in zip(nucleotides, kernel.flatten()):
                    if nucleotide in counts and weight != 0:
                        counts[nucleotide] += abs(weight)
                
                max_nucleotide = max(counts, key=counts.get)
                dna_result += max_nucleotide
            
            result[i_rel, j] = dna_result
    
    return start_row, end_row, result

def apply_dna_convolution(dna_image, kernel_type='A', num_threads=None):
    height, width = dna_image.shape[:2]
    result = np.empty((height, width), dtype='object')
    
    kernel = DNA_KERNELS[kernel_type]
    
    if num_threads is None:
        num_threads = min(8, threading.active_count() * 2)
    
    chunk_size = max(1, height // num_threads)
    chunks = []
    
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunks.append((start_row, end_row, dna_image, kernel, height, width))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for start_row, end_row, chunk_result in executor.map(_process_convolution_chunk, chunks):
            result[start_row:end_row] = chunk_result
    
    return result

def _encode_chunk(chunk_data):
    start_row, end_row, image, rule_number = chunk_data
    height, width = image.shape[:2]
    result = np.empty((end_row - start_row, width), dtype='object')
    
    for i_rel, i in enumerate(range(start_row, end_row)):
        for j in range(width):
            result[i_rel, j] = encode_image_to_dna.encode_pixel_to_dna(image[i, j], rule_number)
    
    return start_row, end_row, result

def parallel_encode_image_to_dna(image, rule_number=1, num_threads=None):
    height, width = image.shape[:2]
    dna_image = np.empty((height, width), dtype='object')
    
    if num_threads is None:
        num_threads = min(8, threading.active_count() * 2)
    
    chunk_size = max(1, height // num_threads)
    chunks = []
    
    for start_row in range(0, height, chunk_size):
        end_row = min(start_row + chunk_size, height)
        chunks.append((start_row, end_row, image, rule_number))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        encode_func = partial(encode_image_to_dna, rule_number=rule_number)
        futures = []
        
        for start_row, end_row, _, _ in chunks:
            chunk_img = image[start_row:end_row]
            future = executor.submit(encode_func, chunk_img)
            futures.append((start_row, end_row, future))
        
        for start_row, end_row, future in futures:
            dna_image[start_row:end_row] = future.result()
    
    return dna_image

def encrypt_image(image, key_params, rule_number=1, num_threads=None):
    if len(image.shape) > 2 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    key = generate_key(gray_image.shape, key_params)
    
    dna_image = encode_image_to_dna(gray_image, rule_number)
    dna_key = encode_image_to_dna(key, rule_number)
    
    dna_xor_result = dna_xor_operation(dna_image, dna_key)
    
    convolved_dna = apply_dna_convolution(dna_xor_result, kernel_type='A', num_threads=num_threads)
    
    encrypted_dna = dna_xor_operation(convolved_dna, dna_key)
    
    encrypted_image = decode_dna_to_image(encrypted_dna, rule_number)
    
    return encrypted_image

def decrypt_image(encrypted_image, key_params, rule_number=1, num_threads=None):
    key = generate_key(encrypted_image.shape, key_params)
    
    encrypted_dna = encode_image_to_dna(encrypted_image, rule_number)
    dna_key = encode_image_to_dna(key, rule_number)
    
    xor_result = dna_xor_operation(encrypted_dna, dna_key)
    
    inv_convolved_dna = apply_dna_convolution(xor_result, kernel_type='T', num_threads=num_threads)
    
    decrypted_dna = dna_xor_operation(inv_convolved_dna, dna_key)
    
    decrypted_image = decode_dna_to_image(decrypted_dna, rule_number)
    
    return decrypted_image