import os
import cv2
import sys
import numpy as np

from hist import histogram_equalization, plot_histogram
from threshold import iterative_three_class_thresholding


def calculate_global_contrast(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        min_val = np.min(img)
        max_val = np.max(img)

        contrast = (max_val - min_val) / 255
        return contrast
    except Exception as e:
        print(f"Error calculating global contrast: {e}")
        return None


def calculate_local_contrast(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        height, width = img.shape
        total_pixels = height * width
        sum_abs_diff = 0.0

        padded_img = cv2.copyMakeBorder(
            img, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)

        for r in range(height):
            for c in range(width):
                current_pixel_value = img[r, c]
                neighborhood = padded_img[r:r+3, c:c+3]

                sum_of_neighbors = np.sum(neighborhood) - current_pixel_value
                avg_neighbor_value = sum_of_neighbors / 8.0

                sum_abs_diff += abs(current_pixel_value - avg_neighbor_value)

        if total_pixels == 0:
            return 0.0

        local_contrast = sum_abs_diff / total_pixels
        return local_contrast
    except Exception as e:
        print(f"Error calculating local contrast: {e}")
        return None


def contrast_lut(image_path, output_image_path, lut_rgb):
    try:
        if not os.path.exists(image_path):
            print(f"err: file not found: {image_path}")
            return

        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"err: image not found or invalid format: {image_path}")
            return

        output_img_color = np.zeros(
            (img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)

        output_img_color[:, :, 0] = lut_rgb[2][img_gray]
        output_img_color[:, :, 1] = lut_rgb[1][img_gray]
        output_img_color[:, :, 2] = lut_rgb[0][img_gray]

        cv2.imwrite(output_image_path, output_img_color)
        print(f"image saved as: {output_image_path}")
    except Exception as e:
        print(f"err: {e}")


def gen_lut():
    lut_b = np.zeros(256, dtype=np.uint8)
    lut_g = np.zeros(256, dtype=np.uint8)
    lut_r = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if 0 <= i <= 31:
            lut_b[i] = round(128 + (127 / 31) * i)
        elif 31 < i <= 95:
            lut_b[i] = 255
        elif 95 < i <= 159:
            lut_b[i] = round(255 - (255 / 64) * (i - 95))
        else:
            lut_b[i] = 0

        if 0 <= i <= 31:
            lut_r[i] = 0
        elif 31 < i <= 95:
            lut_r[i] = round((255 / 64) * (i - 31))
        elif 95 < i <= 159:
            lut_r[i] = 255
        elif 159 < i <= 223:
            lut_r[i] = round(255 - (255 / 64) * (i - 159))
        else:
            lut_r[i] = 0

        if 0 <= i <= 95:
            lut_g[i] = 0
        elif 95 < i <= 159:
            lut_g[i] = round((255 / 64) * (i - 95))
        elif 159 < i <= 223:
            lut_g[i] = 255
        else:
            lut_g[i] = round(255 - (127 / 32) * (i - 223))

    return (lut_r, lut_g, lut_b)


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 contrast.py <PATH_IN> <VARIANT> <PATH_OUT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]
    path_out = sys.argv[3]

    if var == "global":
        result = calculate_global_contrast(path_in)
        print(f"Global contrast: {result}")
    elif var == "local":
        result = calculate_local_contrast(path_in)
        print(f"Local contrast: {result}")
    elif var == "lut":
        lut_rgb = gen_lut()
        contrast_lut(path_in, path_out, lut_rgb)
    elif var == "ct":
        histogram_equalization(path_in, "output/CalunTurynski_equalized.png")
        iterative_three_class_thresholding(
            "output/CalunTurynski_equalized.png", path_out)

        img = cv2.imread(path_out, cv2.IMREAD_GRAYSCALE)
        plot_histogram(img, "Histogram i skumulowany histogram",
                       "output/CalunTurynski_equalized_three_class_hist.png")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
