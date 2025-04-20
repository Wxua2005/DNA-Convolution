# /Users/naren/Desktop/DNA-Convolution/chaotic_maps.py
import numpy as np

def logistic_map(x, r, n_iterations):
    sequence = np.zeros(n_iterations)
    sequence[0] = x
    
    for i in range(1, n_iterations):
        sequence[i] = r * sequence[i-1] * (1 - sequence[i-1])
    
    return sequence

def generate_chaotic_sequence(initial_value, r, size, discard=100):
    total_size = size[0] * size[1] + discard
    sequence = logistic_map(initial_value, r, total_size)

    return np.reshape(sequence[discard:], size)

def generate_key(image_shape, key_params):

    key_sequence = generate_chaotic_sequence(
        key_params['x0'], 
        key_params['r'], 
        image_shape)
    
    key = np.uint8(key_sequence * 255)
    return key