import sys
from PIL import Image
import numpy as np


def resize_nearest_neighbor(image, new_width, new_height):
    original = np.array(image)
    height_in, width_in = original.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    scale_x = width_in / new_width
    scale_y = height_in / new_height

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_old = min(round(x_new * scale_x), width_in - 1)
            y_old = min(round(y_new * scale_y), height_in - 1)
            result[y_new, x_new] = original[y_old, x_old]

    return Image.fromarray(result)


def resize_average_two_horizontal(image, new_width, new_height):
    original = np.array(image)
    height_in, width_in = original.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    scale_x = width_in / new_width
    scale_y = height_in / new_height

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_src = x_new * scale_x
            y_src = int(y_new * scale_y)
            x0 = int(np.floor(x_src))
            x1 = min(x0 + 1, width_in - 1)
            y_src = min(y_src, height_in - 1)

            pixel1 = original[y_src, x0]
            pixel2 = original[y_src, x1]
            avg_pixel = ((pixel1.astype(np.uint16) +
                         pixel2.astype(np.uint16)) // 2).astype(np.uint8)

            result[y_new, x_new] = avg_pixel

    return Image.fromarray(result)


def resize_bilinear(image, new_width, new_height):
    original = np.array(image)
    height_in, width_in = original.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    scale_x = width_in / new_width
    scale_y = height_in / new_height

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_src = x_new * scale_x
            y_src = y_new * scale_y

            x0 = int(np.floor(x_src))
            x1 = min(x0 + 1, width_in - 1)
            y0 = int(np.floor(y_src))
            y1 = min(y0 + 1, height_in - 1)

            dx = x_src - x0
            dy = y_src - y0

            p00 = original[y0, x0].astype(np.float32)
            p01 = original[y0, x1].astype(np.float32)
            p10 = original[y1, x0].astype(np.float32)
            p11 = original[y1, x1].astype(np.float32)

            top = p00 * (1 - dx) + p01 * dx
            bottom = p10 * (1 - dx) + p11 * dx
            pixel = top * (1 - dy) + bottom * dy

            result[y_new, x_new] = np.clip(pixel, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def resize_rgb_min_max_avg(image, new_width, new_height):
    original = np.array(image)
    height_in, width_in = original.shape[:2]
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    scale_x = width_in / new_width
    scale_y = height_in / new_height

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_src = x_new * scale_x
            y_src = y_new * scale_y

            x0 = int(np.floor(x_src))
            x1 = min(x0 + 1, width_in - 1)
            y0 = int(np.floor(y_src))
            y1 = min(y0 + 1, height_in - 1)

            p00 = original[y0, x0]
            p01 = original[y0, x1]
            p10 = original[y1, x0]
            p11 = original[y1, x1]

            pixel_block = np.array([p00, p01, p10, p11], dtype=np.uint16)

            avg_rgb = []
            for channel in range(3):
                values = pixel_block[:, channel]
                avg_val = (np.min(values) + np.max(values)) // 2
                avg_rgb.append(avg_val)

            result[y_new, x_new] = avg_rgb

    return Image.fromarray(result.astype(np.uint8))
