from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
import random
from flask_cors import CORS

encryption_keys = {}
app = Flask(__name__)
CORS(app)
# function to generate chaotic sequence (Logistic Map)
def generate_chaotic_sequence(size, x0):
    r = 3.99
    sequence = []
    x = x0
    for _ in range(size):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence) 

# function to generate Playfair matrix
def generate_playfair_matrix(chaotic_seq):
    values = list(range(256))
    random.Random(chaotic_seq[0]).shuffle(values)
    matrix = np.array(values).reshape((16, 16))
    return matrix

# function for Playfair encryption of a block
def playfair_encrypt_block(block, playfair_matrix):
    encrypted_block = np.zeros_like(block)
    rows, cols = playfair_matrix.shape
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            byte = block[i, j]
            row, col = np.where(playfair_matrix == byte)
            row, col = row[0], col[0]
            encrypted_row = (row + 1) % rows
            encrypted_col = (col + 1) % cols
            encrypted_block[i, j] = playfair_matrix[encrypted_row, encrypted_col]
    return encrypted_block

# function to process image channels (RGB)
def process_channel(channel, playfair_matrix, chaotic_seq):
    height, width = channel.shape
    encrypted_channel = np.zeros_like(channel)
    index = 0

    for i in range(0, height, 16):
        for j in range(0, width, 16):
            block = channel[i:i+16, j:j+16]
            block_height, block_width = block.shape
            padded_block = np.zeros((16, 16), dtype=block.dtype)
            padded_block[:block_height, :block_width] = block

            encrypted_block = playfair_encrypt_block(padded_block, playfair_matrix)
            encrypted_channel[i:i+block_height, j:j+block_width] = encrypted_block[:block_height, :block_width]
    return encrypted_channel

# Function to encrypt the image
def encrypt_image(image, x0, rounds=1):
    b_channel, g_channel, r_channel = cv2.split(image)
    chaotic_seq = generate_chaotic_sequence(256, x0)
    playfair_matrix = generate_playfair_matrix(chaotic_seq)

    for _ in range(rounds):  
        b_channel = process_channel(b_channel, playfair_matrix, chaotic_seq)
        g_channel = process_channel(g_channel, playfair_matrix, chaotic_seq)
        r_channel = process_channel(r_channel, playfair_matrix, chaotic_seq)

    encrypted_image = cv2.merge((b_channel, g_channel, r_channel))
    return encrypted_image

def playfair_decrypt_block(block, playfair_matrix):
    decrypted_block = np.zeros_like(block)
    rows, cols = playfair_matrix.shape
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            byte = block[i, j]
            row, col = np.where(playfair_matrix == byte)
            row, col = row[0], col[0]
            decrypted_row = (row - 1) % rows
            decrypted_col = (col - 1) % cols
            decrypted_block[i, j] = playfair_matrix[decrypted_row, decrypted_col]
    return decrypted_block

def process_channel_decrypt(channel, playfair_matrix):
    height, width = channel.shape
    decrypted_channel = np.zeros_like(channel)

    for i in range(0, height, 16):
        for j in range(0, width, 16):
            block = channel[i:i+16, j:j+16]
            block_height, block_width = block.shape
            padded_block = np.zeros((16, 16), dtype=block.dtype)
            padded_block[:block_height, :block_width] = block

            decrypted_block = playfair_decrypt_block(padded_block, playfair_matrix)
            decrypted_channel[i:i+block_height, j:j+block_width] = decrypted_block[:block_height, :block_width]
    return decrypted_channel

def decrypt_image(image, x0, rounds=1):
    b_channel, g_channel, r_channel = cv2.split(image)
    chaotic_seq = generate_chaotic_sequence(256, x0)
    playfair_matrix = generate_playfair_matrix(chaotic_seq)

    for _ in range(rounds):
        b_channel = process_channel_decrypt(b_channel, playfair_matrix)
        g_channel = process_channel_decrypt(g_channel, playfair_matrix)
        r_channel = process_channel_decrypt(r_channel, playfair_matrix)

    decrypted_image = cv2.merge((b_channel, g_channel, r_channel))
    return decrypted_image

@app.route('/encrypt', methods=['POST'])
def encrypt_image_route():
    file = request.files.get('image')
    if file is None:
        return jsonify({"error": "No image file found in request."}), 400

    # Get the encryption key
    key = request.form.get('key', None)
    
    print("key value"+key)
    # Check if the key is provided and valid
    if not key:
        return jsonify({"error": "No encryption key provided."}), 400

    try:
        x0 = float(key)  # Try to convert the key to a float
    except ValueError:
        return jsonify({"error": "Invalid encryption key. Please provide a numeric key."}), 400
    encryption_keys=x0
    # Save the uploaded image temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

   
    img = cv2.imread(temp_path)

    if img is None:
        return jsonify({"error": "Image not found! Could not load the image."}), 400

    # Encrypt the image
    encrypted_img = encrypt_image(img, x0)

    
    encrypted_path = 'encrypted_image.png'
    cv2.imwrite(encrypted_path, encrypted_img)

   
    return send_file(encrypted_path, mimetype='image/png')


@app.route('/decrypt', methods=['POST'])
def decrypt_image_route():
    file = request.files.get('image')
    if file is None:
        return jsonify({"error": "No image file found in request."}), 400

    # Get the decryption key
    key = request.form.get('key', None)
    print(key)
   
    if not key:
        return jsonify({"error": "No decryption key provided."}), 400

    try:
        x0 = float(key)  
    except ValueError:
        return jsonify({"error": "Invalid decryption key. Please provide a numeric key."}), 400
    
    if encryption_keys== key:
        return jsonify({"error":"Invalid decryption key"})
    # Save the uploaded image temporarily
    temp_path = 'temp_encrypted_image.png'
    file.save(temp_path)

    img = cv2.imread(temp_path)
    if img is None:
        return jsonify({"error": "Image not found! Could not load the image."}), 400

    # Decrypt the image
    decrypted_img = decrypt_image(img, x0)
    decrypted_path = 'decrypted_image.png'
    cv2.imwrite(decrypted_path, decrypted_img)

    return send_file(decrypted_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
