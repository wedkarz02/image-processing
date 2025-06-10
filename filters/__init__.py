import os
import sys
import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift


def sine_window(image_path, output_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        height, width = img.shape
        img_float = img.astype(np.float32)

        n = np.arange(width)
        sine_window_1d = np.sin(np.pi * n / (width - 1))
        windowed_img_float = img_float * sine_window_1d[np.newaxis, :]
        windowed_img_uint8 = np.clip(
            windowed_img_float, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, windowed_img_uint8)
        print(f"image saved as: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def averaging_filter(image_path, output_path, kernel_size=(3, 3)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        filtered_img = cv2.blur(img, kernel_size)

        cv2.imwrite(output_path, filtered_img)
        print(f"image saved as: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def gamma_correction_to_match_original(input_image_path, original_image_path, output_path):
    try:
        original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise FileNotFoundError(
                f"Could not open or find the original image at {original_image_path}")
        original_mean_brightness = np.mean(original_img)
        print(
            f"Original image mean brightness: {original_mean_brightness:.2f}")

        img_to_correct = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if img_to_correct is None:
            raise FileNotFoundError(
                f"Could not open or find the image to correct at {input_image_path}")

        best_gamma = 1.0
        min_diff = float('inf')

        gamma_range = np.arange(0.01, 10.0, 0.01)

        for gamma_val in gamma_range:
            img_normalized = img_to_correct / 255.0
            gamma_corrected_img_normalized = np.power(
                img_normalized, gamma_val)

            gamma_corrected_img_uint8 = np.clip(
                gamma_corrected_img_normalized * 255.0, 0, 255).astype(np.uint8)

            current_mean_brightness = np.mean(gamma_corrected_img_uint8)
            diff = abs(current_mean_brightness - original_mean_brightness)

            if diff < min_diff:
                min_diff = diff
                best_gamma = gamma_val

            if min_diff < 0.5:
                break

        print(
            f"Found optimal gamma: {best_gamma:.2f} with mean brightness difference: {min_diff:.2f}")

        img_normalized = img_to_correct / 255.0
        final_gamma_corrected_img_normalized = np.power(
            img_normalized, best_gamma)
        final_gamma_corrected_img_uint8 = np.clip(
            final_gamma_corrected_img_normalized * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, final_gamma_corrected_img_uint8)
        print(f"Gamma corrected image saved as: {output_path}")

    except Exception as e:
        print(f"Error during gamma correction: {e}")
        sys.exit(1)


def van_cittert_deconvolution(blurred_image, kernel, iterations):
    blurred_image = blurred_image.astype(float)
    M, N = blurred_image.shape

    padded_kernel = np.zeros((M, N))
    k_rows, k_cols = kernel.shape

    padded_kernel[:k_rows, :k_cols] = kernel
    padded_kernel = fftshift(padded_kernel)

    G_prime = fft2(blurred_image)
    H = fft2(padded_kernel)

    G_k = G_prime
    difference_images = []

    for i in range(iterations):
        one_matrix = np.ones_like(H)
        G_k_next = G_prime + (one_matrix - H) * G_k
        G_k = G_k_next

        current_g = np.real(ifft2(G_k))

        difference_image = np.abs(current_g - blurred_image)
        difference_images.append(difference_image)

    deconvolved_image = np.real(ifft2(G_k))
    deconvolved_image = np.clip(deconvolved_image, 0, 1)
    return deconvolved_image, difference_images


def k_trimmed_mean_filter(image_path, k=2, window_size=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not open or find the image at {image_path}")

    if window_size % 2 == 0:
        return None

    rows, cols = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    pad = window_size // 2

    for r in range(rows):
        for c in range(cols):
            r_start = max(0, r - pad)
            r_end = min(rows, r + pad + 1)
            c_start = max(0, c - pad)
            c_end = min(cols, c + pad + 1)

            window = image[r_start:r_end, c_start:c_end].flatten()
            window_sorted = np.sort(window)

            if len(window_sorted) > 2 * k:
                trimmed_values = window_sorted[k:len(window_sorted) - k]
            else:
                if len(window_sorted) > 0:
                    trimmed_values = np.array([np.median(window_sorted)])
                else:
                    trimmed_values = np.array([])

            if len(trimmed_values) > 0:
                filtered_image[r, c] = np.mean(trimmed_values)
            else:
                filtered_image[r, c] = image[r, c]

    return filtered_image


def k_nearest_neighbor_filter(image_path, k=6, window_size=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not open or find the image at {image_path}")

    if window_size % 2 == 0:
        return None

    rows, cols = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    pad = window_size // 2

    for r in range(rows):
        for c in range(cols):
            center_pixel_value = image[r, c]

            r_start = max(0, r - pad)
            r_end = min(rows, r + pad + 1)
            c_start = max(0, c - pad)
            c_end = min(cols, c + pad + 1)

            window_values = []
            for wr in range(r_start, r_end):
                for wc in range(c_start, c_end):
                    window_values.append(image[wr, wc])

            distances_and_values = []
            for val in window_values:
                diff = int(val) - int(center_pixel_value)
                distances_and_values.append((abs(diff), val))

            distances_and_values.sort(key=lambda x: x[0])

            k_nearest_values = []
            for i in range(min(k, len(distances_and_values))):
                k_nearest_values.append(distances_and_values[i][1])

            if len(k_nearest_values) > 0:
                filtered_image[r, c] = np.mean(k_nearest_values)
            else:
                filtered_image[r, c] = center_pixel_value

    return filtered_image


def symmetric_nearest_neighbor_filter(image_path, window_size=3):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not open or find the image at {image_path}")

    if window_size % 2 == 0:
        return None

    rows, cols = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    pad = window_size // 2

    for r in range(rows):
        for c in range(cols):
            center_pixel_value = image[r, c]
            symm_values = []

            for i in range(1, pad + 1):
                if c - i >= 0 and c + i < cols:
                    left_val = image[r, c - i]
                    right_val = image[r, c + i]
                    if abs(int(left_val) - int(center_pixel_value)) < abs(int(right_val) - int(center_pixel_value)):
                        symm_values.append(left_val)
                    else:
                        symm_values.append(right_val)

                if r - i >= 0 and r + i < rows:
                    top_val = image[r - i, c]
                    bottom_val = image[r + i, c]
                    if abs(int(top_val) - int(center_pixel_value)) < abs(int(bottom_val) - int(center_pixel_value)):
                        symm_values.append(top_val)
                    else:
                        symm_values.append(bottom_val)

                if r - i >= 0 and c - i >= 0 and r + i < rows and c + i < cols:
                    val1 = image[r - i, c - i]
                    val2 = image[r + i, c + i]
                    if abs(int(val1) - int(center_pixel_value)) < abs(int(val2) - int(center_pixel_value)):
                        symm_values.append(val1)
                    else:
                        symm_values.append(val2)

                if r - i >= 0 and c + i < cols and r + i < rows and c - i >= 0:
                    val1 = image[r - i, c + i]
                    val2 = image[r + i, c - i]
                    if abs(int(val1) - int(center_pixel_value)) < abs(int(val2) - int(center_pixel_value)):
                        symm_values.append(val1)
                    else:
                        symm_values.append(val2)

            symm_values.append(center_pixel_value)

            if len(symm_values) > 0:
                filtered_image[r, c] = np.mean(symm_values)
            else:
                filtered_image[r, c] = center_pixel_value

    return filtered_image
