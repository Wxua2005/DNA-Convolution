import numpy as np

DNA_ENCODING_RULES = {
    1: {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
    2: {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},
    3: {'00': 'A', '01': 'T', '10': 'C', '11': 'G'},
    4: {'00': 'A', '01': 'T', '10': 'G', '11': 'C'},
    5: {'00': 'C', '01': 'A', '10': 'G', '11': 'T'},
    6: {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
    7: {'00': 'C', '01': 'G', '10': 'A', '11': 'T'},
    8: {'00': 'C', '01': 'T', '10': 'A', '11': 'G'}
}

DNA_DECODING_RULES = {
    rule_num: {v: k for k, v in rule_dict.items()}
    for rule_num, rule_dict in DNA_ENCODING_RULES.items()
}

def decimal_to_binary(decimal, width=8):
    return format(decimal, f'0{width}b')

def binary_to_decimal(binary):
    return int(binary, 2)

def encode_pixel_to_dna(pixel, rule_number=1):
    binary = decimal_to_binary(pixel)
    dna_sequence = ''
    rule = DNA_ENCODING_RULES[rule_number]
    
    for i in range(0, len(binary), 2):
        dna_sequence += rule[binary[i:i+2]]
    
    return dna_sequence

def decode_dna_to_pixel(dna_sequence, rule_number=1):
    binary = ''
    rule = DNA_DECODING_RULES[rule_number]
    
    for nucleotide in dna_sequence:
        binary += rule[nucleotide]
    
    return binary_to_decimal(binary)

def encode_image_to_dna(image, rule_number=1):
    height, width = image.shape[:2]
    dna_image = np.empty((height, width), dtype='object')
    
    for i in range(height):
        for j in range(width):
            dna_image[i, j] = encode_pixel_to_dna(image[i, j], rule_number)
    
    return dna_image

def decode_dna_to_image(dna_image, rule_number=1):
    height, width = dna_image.shape[:2]
    image = np.empty((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            image[i, j] = decode_dna_to_pixel(dna_image[i, j], rule_number)
    
    return image

DNA_XOR = {
    ('A', 'A'): 'A', ('A', 'C'): 'C', ('A', 'G'): 'G', ('A', 'T'): 'T',
    ('C', 'A'): 'C', ('C', 'C'): 'A', ('C', 'G'): 'T', ('C', 'T'): 'G',
    ('G', 'A'): 'G', ('G', 'C'): 'T', ('G', 'G'): 'A', ('G', 'T'): 'C',
    ('T', 'A'): 'T', ('T', 'C'): 'G', ('T', 'G'): 'C', ('T', 'T'): 'A'
}

def dna_xor(dna_seq1, dna_seq2):
    result = ''
    for n1, n2 in zip(dna_seq1, dna_seq2):
        result += DNA_XOR[(n1, n2)]
    return result

def dna_xor_operation(dna_image1, dna_image2):
    height, width = dna_image1.shape[:2]
    result = np.empty((height, width), dtype='object')
    
    for i in range(height):
        for j in range(width):
            result[i, j] = dna_xor(dna_image1[i, j], dna_image2[i, j])
    
    return result