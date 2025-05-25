import cv2
import sys

import numpy as np


def otsu_thresholding(image_path, output_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"err: image not found: {image_path}")
            return

        ret, otsu_thresholded = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"T Otsu threshold: {ret}")

        cv2.imwrite(output_path, otsu_thresholded)
        print(f"image saved at: {output_path}")

    except Exception as e:
        print(f"err: {e}")


def iterative_three_class_thresholding(image_path, output_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"err: image not found: {image_path}")
            return

        T1_old = 80
        T2_old = 170

        delta_T = float('inf')

        while delta_T >= 2:
            class1_pixels = img[img < T1_old]
            class2_pixels = img[(img >= T1_old) & (img < T2_old)]
            class3_pixels = img[img >= T2_old]

            m1 = np.mean(class1_pixels) if class1_pixels.size > 0 else 0
            m2 = np.mean(class2_pixels) if class2_pixels.size > 0 else 0
            m3 = np.mean(class3_pixels) if class3_pixels.size > 0 else 0

            T1_new = (m1 + m2) / 2
            T2_new = (m2 + m3) / 2

            delta_T = max(abs(T1_new - T1_old), abs(T2_new - T2_old))

            print(
                f"  T1_old={T1_old:.2f}, T2_old={T2_old:.2f}, T1_new={T1_new:.2f}, T2_new={T2_new:.2f}, delta={delta_T:.2f}")

            T1_old = T1_new
            T2_old = T2_new

        three_class_thresholded = np.zeros_like(img)
        three_class_thresholded[img < T1_old] = 0
        three_class_thresholded[(img >= T1_old) & (
            img < T2_old)] = 127
        three_class_thresholded[img >= T2_old] = 255

        print(f"thresholds T1: {T1_old:.2f}, T2: {T2_old:.2f}")

        cv2.imwrite(output_path, three_class_thresholded)
        print(
            f"image saved at: {output_path}")

    except Exception as e:
        print(f"err: {e}")


def local_otsu_thresholding(image_path, output_path, window_size=11):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"err: image not found: {image_path}")
            return

        rows, cols = img.shape
        half_window = window_size // 2
        output_img = np.zeros_like(img)

        padded_img = cv2.copyMakeBorder(img, half_window, half_window,
                                        half_window, half_window,
                                        cv2.BORDER_REFLECT)

        for r in range(rows):
            for c in range(cols):
                window = padded_img[r: r + window_size,
                                    c: c + window_size]

                ret, _ = cv2.threshold(
                    window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                if img[r, c] > ret:
                    output_img[r, c] = 255
                else:
                    output_img[r, c] = 0

        cv2.imwrite(output_path, output_img)
        print(
            f"image saved at: {output_path}")

    except Exception as e:
        print(f"err: {e}")


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 threshold.py <PATH_IN> <VARIANT> <PATH_OUT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]
    path_out = sys.argv[3]

    if var == "a":
        otsu_thresholding(path_in, path_out)
    elif var == "b":
        iterative_three_class_thresholding(path_in, path_out)
    elif var == "c":
        local_otsu_thresholding(path_in, path_out)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
