import cv2
import sys
import numpy as np


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


def main():
    if len(sys.argv) != 3:
        print("USAGE: python3 contrast.py <PATH_IN> <VARIANT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]

    if var == "global":
        result = calculate_global_contrast(path_in)
        print(f"Global contrast: {result}")
    elif var == "local":
        result = calculate_local_contrast(path_in)
        print(f"Local contrast: {result}")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
