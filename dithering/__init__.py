import sys
from PIL import Image
import numpy as np


def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0]])

    prev_matrix = generate_bayer_matrix(n // 2)

    top_left = 4 * prev_matrix
    top_right = 4 * prev_matrix + 2
    bottom_left = 4 * prev_matrix + 3
    bottom_right = 4 * prev_matrix + 1

    return np.block([[top_left, top_right],
                     [bottom_left, bottom_right]])


def quantization(pixel_value, rules):
    pixel_value = np.clip(pixel_value, 0, 255)
    quantized_value = 0
    for threshold, palette_value in rules:
        if pixel_value < threshold:
            quantized_value = palette_value
            break
    return quantized_value


def floyd_steinberg(img, rules):
    pixels = np.array(img, dtype=np.float32)
    width, height = img.size

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = quantization(old_pixel, rules)
            error = old_pixel - new_pixel
            pixels[y, x] = new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 3 / 16
                pixels[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    pixels[y + 1, x + 1] += error * 1 / 16

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def jarvis_judice_ninke(img, rules):
    pixels = np.array(img, dtype=np.float32)
    width, height = img.size

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]

            new_pixel = quantization(old_pixel, rules)

            error = old_pixel - new_pixel
            pixels[y, x] = new_pixel

            if x + 1 < width:
                pixels[y, x + 1] += error * 7 / 48
            if x + 2 < width:
                pixels[y, x + 2] += error * 5 / 48

            if y + 1 < height:
                if x - 2 >= 0:
                    pixels[y + 1, x - 2] += error * 3 / 48
                if x - 1 >= 0:
                    pixels[y + 1, x - 1] += error * 5 / 48
                pixels[y + 1, x] += error * 7 / 48
                if x + 1 < width:
                    pixels[y + 1, x + 1] += error * \
                        6 / 48
                if x + 2 < width:
                    pixels[y + 1, x + 2] += error * 3 / 48

            if y + 2 < height:
                if x - 2 >= 0:
                    pixels[y + 2, x - 2] += error * 1 / 48
                if x - 1 >= 0:
                    pixels[y + 2, x - 1] += error * 3 / 48
                pixels[y + 2, x] += error * 5 / 48
                if x + 1 < width:
                    pixels[y + 2, x + 1] += error * 3 / 48
                if x + 2 < width:
                    pixels[y + 2, x + 2] += error * 1 / 48

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def ordered_dithering(img, dithering_matrix):
    pixels = np.array(img, dtype=np.float32)
    width, height = img.size

    matrix_size = dithering_matrix.shape[0]
    num_shades = matrix_size ** 2 + 1

    if num_shades <= 1:
        palette = [0]
    else:
        palette = [int(i * (255 / (num_shades - 1)))
                   for i in range(num_shades)]

    max_val_in_matrix = dithering_matrix.max()
    normalized_dither_matrix = dithering_matrix / \
        (max_val_in_matrix + 1)

    for y in range(height):
        for x in range(width):
            original_pixel_value = pixels[y, x]
            dither_val = normalized_dither_matrix[y %
                                                  matrix_size, x % matrix_size]
            pixel_scaled_to_palette_range = original_pixel_value * \
                (num_shades - 1) / 255.0

            dither_noise = dither_val - 0.5
            new_index = int(
                np.floor(pixel_scaled_to_palette_range + dither_noise + 0.5))

            new_index = np.clip(new_index, 0, num_shades - 1)
            final_quantized_value = palette[new_index]
            pixels[y, x] = final_quantized_value

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def ordered_dithering_bayer_8x8(img, rules):
    pixels = np.array(img, dtype=np.float32)
    width, height = img.size

    bayer_matrix_size = 8
    bayer_matrix = generate_bayer_matrix(8)

    normalized_bayer_matrix = bayer_matrix / \
        (bayer_matrix_size**2)

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            dither_val = normalized_bayer_matrix[y %
                                                 bayer_matrix_size, x % bayer_matrix_size]

            adjusted_pixel_value = old_pixel + (dither_val - 0.5) * 256.0
            new_pixel = quantization(adjusted_pixel_value, rules)
            pixels[y, x] = new_pixel

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
