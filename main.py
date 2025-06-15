import argparse
import numpy as np
import cv2
import sys
import os
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import pandas as pd


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
                    pixels[y + 1, x + 1] += error * 6 / 48
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
    normalized_dither_matrix = dithering_matrix / (max_val_in_matrix + 1)

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
    bayer_matrix = generate_bayer_matrix(bayer_matrix_size)
    normalized_bayer_matrix = bayer_matrix / (bayer_matrix_size**2)

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            dither_val = normalized_bayer_matrix[y %
                                                 bayer_matrix_size, x % bayer_matrix_size]
            adjusted_pixel_value = old_pixel + (dither_val - 0.5) * 256.0
            new_pixel = quantization(adjusted_pixel_value, rules)
            pixels[y, x] = new_pixel

    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


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
        print(f"Image processed and saved as: {output_path}")

    except Exception as e:
        print(f"Error during sine window application: {e}", file=sys.stderr)
        sys.exit(1)


def averaging_filter(image_path, output_path, kernel_size=(3, 3)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        filtered_img = cv2.blur(img, kernel_size)

        cv2.imwrite(output_path, filtered_img)
        print(f"Image processed and saved as: {output_path}")
    except Exception as e:
        print(
            f"Error during averaging filter application: {e}", file=sys.stderr)
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
        print(f"Error during gamma correction: {e}", file=sys.stderr)
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


def k_trimmed_mean_filter(image_path, output_path, k=2, window_size=3):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be odd for the K-trimmed mean filter.")

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

        cv2.imwrite(output_path, filtered_image)
        print(f"K-trimmed mean filtered image saved as: {output_path}")

    except Exception as e:
        print(f"Error applying K-trimmed mean filter: {e}", file=sys.stderr)
        sys.exit(1)


def k_nearest_neighbor_filter(image_path, output_path, k=6, window_size=3):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be odd for the K-nearest neighbor filter.")

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

        cv2.imwrite(output_path, filtered_image)
        print(f"K-nearest neighbor filtered image saved as: {output_path}")

    except Exception as e:
        print(
            f"Error applying K-nearest neighbor filter: {e}", file=sys.stderr)
        sys.exit(1)


def symmetric_nearest_neighbor_filter(image_path, output_path, window_size=3):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be odd for the Symmetric Nearest Neighbor filter.")

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

        cv2.imwrite(output_path, filtered_image)
        print(
            f"Symmetric Nearest Neighbor filtered image saved as: {output_path}")

    except Exception as e:
        print(
            f"Error applying Symmetric Nearest Neighbor filter: {e}", file=sys.stderr)
        sys.exit(1)


def calculate_global_contrast(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        min_val = np.min(img)
        max_val = np.max(img)

        contrast = (max_val - min_val) / 255.0
        return contrast
    except Exception as e:
        print(f"Error calculating global contrast: {e}", file=sys.stderr)
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
        print(f"Error calculating local contrast: {e}", file=sys.stderr)
        return None


def contrast_lut(image_path, output_image_path, lut_rgb):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            raise ValueError(
                f"Image not found or invalid format: {image_path}")

        output_img_color = np.zeros(
            (img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)

        output_img_color[:, :, 0] = lut_rgb[0][img_gray]
        output_img_color[:, :, 1] = lut_rgb[1][img_gray]
        output_img_color[:, :, 2] = lut_rgb[2][img_gray]

        cv2.imwrite(output_image_path, output_img_color)
        print(f"Image processed with LUT and saved as: {output_image_path}")
    except Exception as e:
        print(f"Error applying contrast LUT: {e}", file=sys.stderr)
        sys.exit(1)


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


def process_image_fourier(image_path, output_plot_prefix="fourier_analysis"):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_float = img_gray.astype(np.float32)

        rows, cols = img_float.shape

        f_transform = np.fft.fft2(img_float)
        f_transform_shifted = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1e-10)

        radius_low = min(rows, cols) // 10
        radius_high = min(rows, cols) // 4

        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[-center_row:rows -
                        center_row, -center_col:cols - center_col]

        mask_low = (x**2 + y**2 <= radius_low**2).astype(np.float32)

        mask_high = (x**2 + y**2 <= radius_high**2).astype(np.float32)

        f_low_freq_shifted = f_transform_shifted * mask_low
        f_high_freq_shifted = f_transform_shifted * mask_high

        img_reconstructed_low_freq = np.abs(
            np.fft.ifft2(np.fft.ifftshift(f_low_freq_shifted)))
        img_reconstructed_high_freq = np.abs(
            np.fft.ifft2(np.fft.ifftshift(f_high_freq_shifted)))

        img_reconstructed_low_freq = cv2.normalize(
            img_reconstructed_low_freq, None, 0, 255, cv2.NORM_MINMAX)
        img_reconstructed_low_freq = img_reconstructed_low_freq.astype(
            np.uint8)

        img_reconstructed_high_freq = cv2.normalize(
            img_reconstructed_high_freq, None, 0, 255, cv2.NORM_MINMAX)
        img_reconstructed_high_freq = img_reconstructed_high_freq.astype(
            np.uint8)

        profile_row_index = rows // 2
        profile_original = img_float[profile_row_index, :]
        profile_low_freq = img_reconstructed_low_freq[profile_row_index, :]
        profile_high_freq = img_reconstructed_high_freq[profile_row_index, :]

        plt.figure(figsize=(15, 12))

        plt.subplot(3, 3, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(3, 3, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum (FFT)')
        plt.axis('off')

        plt.subplot(3, 3, 3)
        plt.imshow(mask_low, cmap='gray')
        plt.title(f'Low-Pass Mask (R={radius_low})')
        plt.axis('off')

        plt.subplot(3, 3, 4)
        plt.imshow(img_reconstructed_low_freq, cmap='gray')
        plt.title('Reconstruction (Low Freq.)')
        plt.axis('off')

        plt.subplot(3, 3, 5)
        plt.imshow(mask_high, cmap='gray')
        plt.title(f'Wider Low-Pass Mask (R={radius_high})')
        plt.axis('off')

        plt.subplot(3, 3, 6)
        plt.imshow(img_reconstructed_high_freq, cmap='gray')
        plt.title('Reconstruction (Wider Freq.)')
        plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.plot(profile_original, label='Original Profile', color='blue')
        plt.plot(profile_low_freq,
                 label=f'Reconstruction Profile (Low Freq. R={radius_low})', color='red', linestyle='--')
        plt.plot(profile_high_freq,
                 label=f'Reconstruction Profile (Wider Freq. R={radius_high})', color='green', linestyle=':')
        plt.title('Horizontal Intensity Profiles')
        plt.ylabel('Pixel Intensity')
        plt.xlabel('Pixel Position')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = f"{output_plot_prefix}_fourier_analysis.png"
        plt.savefig(plot_path)
        print(f"Fourier analysis plot saved as: {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Error during Fourier analysis: {e}", file=sys.stderr)
        sys.exit(1)


def plot_histogram(image_data, title, output_plot_path):
    hist = cv2.calcHist([image_data], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max() if cdf.max() > 0 else cdf

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Pixel Brightness Level')
    plt.ylabel('Number of Pixels / Normalized Cumulative Pixels')

    plt.plot(hist, color='blue', label='Histogram')
    plt.fill_between(range(256), hist, color='lightblue', alpha=0.5)

    plt.plot(cdf_normalized, color='red',
             label='Normalized Cumulative Histogram')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 255])
    plt.tight_layout()

    plt.savefig(output_plot_path)
    print(f"Plot '{title}' saved as: {output_plot_path}")
    plt.close()


def histogram_equalization(image_path, output_image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(
                f"Image not found or invalid format: {image_path}")

        equalized_img = cv2.equalizeHist(img)

        cv2.imwrite(output_image_path, equalized_img)
        print(f"Equalized image saved as: {output_image_path}")

        output_dir = os.path.dirname(output_image_path)
        output_filename_base = os.path.splitext(
            os.path.basename(output_image_path))[0]
        output_hist_plot_path = os.path.join(
            output_dir, f"{output_filename_base}_hist.png")
        plot_histogram(
            equalized_img, 'Histogram and Normalized CDF of Equalized Image', output_hist_plot_path)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf, 0)
        if cdf_m.count() > 0:
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
        else:
            cdf_final = np.zeros_like(cdf, dtype='uint8')

        g_values = [10, 15, 20]

        print("\nOutput grayscale values H_equal(g) for given g (before cv2 rounding):")
        for g in g_values:
            if 0 <= g <= 255:
                h_equal_g = cdf_final[g]
                print(f"  for g = {g}: H_equal(g) = {h_equal_g}")
            else:
                print(f"  Value g={g} is out of range [0, 255].")

    except Exception as e:
        print(f"Error during histogram equalization: {e}", file=sys.stderr)
        sys.exit(1)


def hyperbolic_histogram_equalization(image_path, output_image_path, alpha=-1/3):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(
                f"Image not found or invalid format: {image_path}")

        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        cdf = hist.cumsum()

        cdf_normalized = cdf / float(img.size)
        cdf_normalized[cdf_normalized == 0] = np.finfo(float).eps

        L_minus_1 = 255.0

        transformed_cdf_vals = np.sign(alpha) * np.power(cdf_normalized, alpha)

        min_val = np.min(transformed_cdf_vals)
        max_val = np.max(transformed_cdf_vals)

        if (max_val - min_val) == 0:
            map_function = np.zeros_like(transformed_cdf_vals)
        else:
            map_function = L_minus_1 * \
                (transformed_cdf_vals - min_val) / (max_val - min_val)

        map_function = np.round(map_function).astype('uint8')
        hyper_equalized_img = map_function[img]

        cv2.imwrite(output_image_path, hyper_equalized_img)
        print(
            f"Hyperbolic equalized image (alpha={alpha}) saved as: {output_image_path}")

        output_dir = os.path.dirname(output_image_path)
        output_filename_base = os.path.splitext(
            os.path.basename(output_image_path))[0]
        output_hist_plot_path = os.path.join(
            output_dir, f"{output_filename_base}_hist.png")
        plot_histogram(hyper_equalized_img,
                       f'Histogram and Normalized CDF of Hyperbolic Equalized Image (alpha={alpha})', output_hist_plot_path)

        g_values = [10, 15, 20]

        print(
            f"\nOutput grayscale values H_hyper(g) for given g (alpha={alpha}):")
        for g in g_values:
            if 0 <= g <= 255:
                h_hyper_g = map_function[g]
                print(f"  for g = {g}: H_hyper(g) = {h_hyper_g}")
            else:
                print(f"  Value g={g} is out of range [0, 255].")

    except Exception as e:
        print(
            f"Error during hyperbolic histogram equalization: {e}", file=sys.stderr)
        sys.exit(1)


def pointillism_effect(image_path, output_path, radius):
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        output = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(output)

        step = radius

        for y in range(0, height, step):
            for x in range(0, width, step):

                dx = random.randint(0, min(radius, width - x - 1))
                dy = random.randint(0, min(radius, height - y - 1))
                px = x + dx
                py = y + dy

                color = image.getpixel((px, py))

                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=color,
                    outline=None
                )

        output.save(output_path)
        print(f"Pointillism effect applied and saved as: {output_path}")

    except Exception as e:
        print(f"Error applying pointillism effect: {e}", file=sys.stderr)
        sys.exit(1)


def plot_profile_with_sampling(csv_file_path, sampling_frequency_pixels=50, output_filename="profile_with_sampling.png"):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: '{csv_file_path}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    distance_col = 'Distance_(pixels)'
    gray_value_col = 'Gray_Value'

    if distance_col not in df.columns or gray_value_col not in df.columns:
        print(
            f"Error: columns '{distance_col}' or '{gray_value_col}' not found in '{csv_file_path}'", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(12, 7))
    plt.plot(df[distance_col], df[gray_value_col], color='black',
             linewidth=1.5, label='Original Line Profile')

    sampled_indices = np.arange(0, len(df), sampling_frequency_pixels)
    sampled_distance = df[distance_col].iloc[sampled_indices]
    sampled_gray_value = df[gray_value_col].iloc[sampled_indices]

    plt.scatter(sampled_distance, sampled_gray_value, color='red', marker='o', s=50,
                zorder=5, label=f'Sampling Points (every {sampling_frequency_pixels} px)')

    for dist, val in zip(sampled_distance, sampled_gray_value):
        plt.axvline(x=dist, color='gray', linestyle='--',
                    linewidth=0.8, alpha=0.6)

    plt.title(
        'Image Line Profile with Sampling Points', fontsize=16)
    plt.xlabel('Distance (pixels)', fontsize=14)
    plt.ylabel('Gray Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved as: '{output_filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
        sys.exit(1)
    plt.close()


def zhang_suen_thinning(image_path, output_path):
    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(
                f"Could not open or find the image at {image_path}")

        _, img_binary_255 = cv2.threshold(
            img_gray, 128, 255, cv2.THRESH_BINARY_INV)
        img_binary_01 = (img_binary_255 == 255).astype(int)

        img = np.pad(img_binary_01, pad_width=1,
                     mode='constant', constant_values=0)

        rows, cols = img.shape
        changed = True
        while changed:
            changed = False

            pixels_to_delete_step1 = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if img[r, c] == 1:

                        p = [0] * 10
                        p[2] = img[r, c+1]
                        p[3] = img[r+1, c+1]
                        p[4] = img[r+1, c]
                        p[5] = img[r+1, c-1]
                        p[6] = img[r, c-1]
                        p[7] = img[r-1, c-1]
                        p[8] = img[r-1, c]
                        p[9] = img[r-1, c+1]

                        B_P1 = sum(p[2:])

                        neighbors_cycle = p[2:] + [p[2]]
                        A_P1 = 0
                        for i in range(8):
                            if neighbors_cycle[i] == 0 and neighbors_cycle[i+1] == 1:
                                A_P1 += 1

                        cond1 = (2 <= B_P1 <= 6)
                        cond2 = (A_P1 == 1)
                        cond3 = (p[2] * p[4] * p[8] == 0)
                        cond4 = (p[2] * p[6] * p[8] == 0)

                        if cond1 and cond2 and cond3 and cond4:
                            pixels_to_delete_step1.append((r, c))

            for r, c in pixels_to_delete_step1:
                if img[r, c] == 1:
                    img[r, c] = 0
                    changed = True

            pixels_to_delete_step2 = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if img[r, c] == 1:
                        p = [0] * 10
                        p[2] = img[r, c+1]
                        p[3] = img[r+1, c+1]
                        p[4] = img[r+1, c]
                        p[5] = img[r+1, c-1]
                        p[6] = img[r, c-1]
                        p[7] = img[r-1, c-1]
                        p[8] = img[r-1, c]
                        p[9] = img[r-1, c+1]

                        B_P1 = sum(p[2:])
                        neighbors_cycle = p[2:] + [p[2]]
                        A_P1 = 0
                        for i in range(8):
                            if neighbors_cycle[i] == 0 and neighbors_cycle[i+1] == 1:
                                A_P1 += 1

                        cond1 = (2 <= B_P1 <= 6)
                        cond2 = (A_P1 == 1)
                        cond3 = (p[2] * p[4] * p[6] == 0)
                        cond4 = (p[4] * p[6] * p[8] == 0)

                        if cond1 and cond2 and cond3 and cond4:
                            pixels_to_delete_step2.append((r, c))

            for r, c in pixels_to_delete_step2:
                if img[r, c] == 1:
                    img[r, c] = 0
                    changed = True

        skeleton_01_no_pad = img[1:-1, 1:-1]
        skeleton_255_output = (1 - skeleton_01_no_pad) * 255

        cv2.imwrite(output_path, skeleton_255_output.astype(np.uint8))
        print(f"Thinned image saved as: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during Zhang-Suen thinning: {e}", file=sys.stderr)
        sys.exit(1)


def otsu_thresholding(image_path, output_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        ret, otsu_thresholded = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"Otsu's threshold value: {ret:.2f}")

        cv2.imwrite(output_path, otsu_thresholded)
        print(f"Thresholded image saved at: {output_path}")

    except Exception as e:
        print(f"Error during Otsu thresholding: {e}", file=sys.stderr)
        sys.exit(1)


def iterative_three_class_thresholding(image_path, output_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        T1_old = 80
        T2_old = 170

        delta_T = float('inf')
        iteration = 0
        max_iterations = 100

        print("Iterative Three-Class Thresholding Progress:")
        while delta_T >= 2 and iteration < max_iterations:
            iteration += 1
            class1_pixels = img[img < T1_old]
            class2_pixels = img[(img >= T1_old) & (img < T2_old)]
            class3_pixels = img[img >= T2_old]

            m1 = np.mean(class1_pixels) if class1_pixels.size > 0 else T1_old
            m2 = np.mean(class2_pixels) if class2_pixels.size > 0 else (
                T1_old + T2_old) / 2
            m3 = np.mean(class3_pixels) if class3_pixels.size > 0 else T2_old

            T1_new = (m1 + m2) / 2
            T2_new = (m2 + m3) / 2

            delta_T = max(abs(T1_new - T1_old), abs(T2_new - T2_old))

            print(
                f"  Iteration {iteration}: T1_old={T1_old:.2f}, T2_old={T2_old:.2f}, T1_new={T1_new:.2f}, T2_new={T2_new:.2f}, delta={delta_T:.2f}")

            T1_old = T1_new
            T2_old = T2_new

        if iteration >= max_iterations:
            print(
                f"Warning: Reached maximum iterations ({max_iterations}) without full convergence.")

        three_class_thresholded = np.zeros_like(img)
        three_class_thresholded[img < T1_old] = 0
        three_class_thresholded[(img >= T1_old) & (img < T2_old)] = 127
        three_class_thresholded[img >= T2_old] = 255

        print(f"Final thresholds: T1: {T1_old:.2f}, T2: {T2_old:.2f}")

        cv2.imwrite(output_path, three_class_thresholded)
        print(
            f"Iterative three-class thresholded image saved at: {output_path}")

    except Exception as e:
        print(
            f"Error during iterative three-class thresholding: {e}", file=sys.stderr)
        sys.exit(1)


def local_otsu_thresholding(image_path, output_path, window_size=11):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if window_size % 2 == 0:
            raise ValueError(
                "Window size must be odd for local Otsu thresholding.")

        rows, cols = img.shape
        half_window = window_size // 2
        output_img = np.zeros_like(img)

        padded_img = cv2.copyMakeBorder(img, half_window, half_window,
                                        half_window, half_window,
                                        cv2.BORDER_REFLECT)

        print(
            f"Applying local Otsu thresholding with window size {window_size}x{window_size}...")
        for r in range(rows):
            for c in range(cols):

                window = padded_img[r: r + window_size,
                                    c: c + window_size]

                if window.size > 0:
                    ret, _ = cv2.threshold(
                        window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    ret = 128

                if img[r, c] > ret:
                    output_img[r, c] = 255
                else:
                    output_img[r, c] = 0

            if rows > 100 and r % (rows // 10) == 0:
                print(f"  Processing row {r}/{rows}...")

        cv2.imwrite(output_path, output_img)
        print(f"Local Otsu thresholded image saved at: {output_path}")

    except Exception as e:
        print(f"Error during local Otsu thresholding: {e}", file=sys.stderr)
        sys.exit(1)


def resize_nearest_neighbor(image_path, output_path, new_width, new_height):
    try:
        image = Image.open(image_path).convert("RGB")
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

        Image.fromarray(result).save(output_path)
        print(f"Image resized (Nearest Neighbor) and saved as: {output_path}")

    except Exception as e:
        print(f"Error during Nearest Neighbor resize: {e}", file=sys.stderr)
        sys.exit(1)


def resize_average_two_horizontal(image_path, output_path, new_width, new_height):
    try:
        image = Image.open(image_path).convert("RGB")
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

        Image.fromarray(result).save(output_path)
        print(
            f"Image resized (Average Two Horizontal) and saved as: {output_path}")

    except Exception as e:
        print(
            f"Error during Average Two Horizontal resize: {e}", file=sys.stderr)
        sys.exit(1)


def resize_bilinear(image_path, output_path, new_width, new_height):
    try:
        image = Image.open(image_path).convert("RGB")
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

        Image.fromarray(result).save(output_path)
        print(f"Image resized (Bilinear) and saved as: {output_path}")

    except Exception as e:
        print(f"Error during Bilinear resize: {e}", file=sys.stderr)
        sys.exit(1)


def resize_rgb_min_max_avg(image_path, output_path, new_width, new_height):
    try:
        image = Image.open(image_path).convert("RGB")
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

        Image.fromarray(result).save(output_path)
        print(
            f"Image resized (RGB Min/Max Average) and saved as: {output_path}")

    except Exception as e:
        print(f"Error during RGB Min/Max Average resize: {e}", file=sys.stderr)
        sys.exit(1)


def generate_basic_lut(lut_type):
    identity_lut = np.arange(256, dtype=np.uint8)
    if lut_type == "identity":
        lut = identity_lut
    elif lut_type == "inverse":
        lut = 255 - identity_lut
    elif lut_type == "high-contrast":
        lut = np.array([int(255 * (((i / 255.0) - 0.5) * 2 + 0.5))
                       for i in range(256)], dtype=np.uint8)
        lut = np.clip(lut, 0, 255)
    elif lut_type == "low-contrast":
        lut = np.array([int(255 * (((i / 255.0) - 0.5) * 0.5 + 0.5))
                       for i in range(256)], dtype=np.uint8)
        lut = np.clip(lut, 0, 255)
    else:
        raise ValueError(
            f"Unknown LUT type: '{lut_type}'. Choose from 'identity', 'inverse', 'high-contrast', 'low-contrast'.")

    return [lut, lut, lut]


def parse_rules(rules_str):
    rules = []
    try:
        pairs = rules_str.split(',')
        for pair in pairs:
            threshold_str, value_str = pair.split(':')
            threshold = int(threshold_str)
            value = int(value_str)
            rules.append((threshold, value))
        rules.sort(key=lambda x: x[0])
    except ValueError:
        raise ValueError(
            "Invalid rules format. Expected 'THRESHOLD1:VALUE1,THRESHOLD2:VALUE2,...' with integer values.")
    return rules


def parse_kernel(kernel_str):
    try:
        rows = kernel_str.split(';')
        kernel_list = []
        for row in rows:
            kernel_list.append([float(val) for val in row.split(',')])
        return np.array(kernel_list)
    except ValueError:
        raise ValueError(
            "Invalid kernel format. Expected 'V1,V2;V3,V4,...' with float values.")


def main():
    parser = argparse.ArgumentParser(
        description="Apply various image processing and analysis techniques.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available image processing commands. Use '[command] --help' for command-specific options."
    )

    fs_parser = subparsers.add_parser("floyd-steinberg", help="Apply Floyd-Steinberg error diffusion dithering.",
                                      description="Applies Floyd-Steinberg error diffusion dithering to a grayscale image.\nRequires quantization rules.")
    fs_parser.add_argument("input_image", help="Path to the input image file.")
    fs_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    fs_parser.add_argument("--rules", type=str, required=True,
                           help="Quantization rules as 'THRESHOLD1:VALUE1,THRESHOLD2:VALUE2,...'. Example: '128:0,256:255'.")

    jjn_parser = subparsers.add_parser("jarvis-judice-ninke", help="Apply Jarvis-Judice-Ninke error diffusion dithering.",
                                       description="Applies Jarvis-Judice-Ninke error diffusion dithering to a grayscale image.\nRequires quantization rules.")
    jjn_parser.add_argument(
        "input_image", help="Path to the input image file.")
    jjn_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    jjn_parser.add_argument("--rules", type=str, required=True,
                            help="Quantization rules as 'THRESHOLD1:VALUE1,THRESHOLD2:VALUE2,...'.")

    od_parser = subparsers.add_parser("ordered-dithering", help="Apply ordered dithering using a generated Bayer matrix.",
                                      description="Applies ordered dithering to a grayscale image using a Bayer matrix of specified size.\nThe palette is automatically derived from the matrix size.")
    od_parser.add_argument("input_image", help="Path to the input image file.")
    od_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    od_parser.add_argument("--bayer-n", type=int, default=8,
                           help="Size 'n' for the Bayer matrix (must be a power of 2, e.g., 2, 4, 8, 16). Default is 8.")

    od8x8_parser = subparsers.add_parser("ordered-dithering-bayer-8x8", help="Apply ordered dithering using a fixed 8x8 Bayer matrix with explicit rules.",
                                         description="Applies ordered dithering to a grayscale image using a fixed 8x8 Bayer matrix.\nRequires explicit quantization rules.")
    od8x8_parser.add_argument(
        "input_image", help="Path to the input image file.")
    od8x8_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    od8x8_parser.add_argument("--rules", type=str, required=True,
                              help="Quantization rules as 'THRESHOLD1:VALUE1,THRESHOLD2:VALUE2,...'.")

    sw_parser = subparsers.add_parser("sine-window", help="Apply a sine window filter.",
                                      description="Applies a sine window function to the image horizontally, enhancing contrast towards the center.")
    sw_parser.add_argument("input_image", help="Path to the input image file.")
    sw_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")

    af_parser = subparsers.add_parser("averaging-filter", help="Apply an averaging (blur) filter.",
                                      description="Applies an averaging (blur) filter to a grayscale image.")
    af_parser.add_argument("input_image", help="Path to the input image file.")
    af_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    af_parser.add_argument("--kernel-size", type=str, default="3x3",
                           help="Kernel size as 'WxH' (e.g., '3x3', '5x5'). Default is 3x3.")

    ktm_parser = subparsers.add_parser("k-trimmed-mean", help="Apply K-trimmed mean filter.",
                                       description="Applies a K-trimmed mean filter to reduce noise while preserving edges.\nTrims 'k' lowest and highest values from the window before averaging.")
    ktm_parser.add_argument(
        "input_image", help="Path to the input image file.")
    ktm_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    ktm_parser.add_argument("--k", type=int, default=2,
                            help="Number of values to trim from each end of the sorted window. Default is 2.")
    ktm_parser.add_argument("--window-size", type=int, default=3,
                            help="Size of the square window (e.g., 3 for 3x3). Must be odd. Default is 3.")

    knn_parser = subparsers.add_parser("k-nearest-neighbor", help="Apply K-nearest neighbor filter.",
                                       description="Applies a K-nearest neighbor filter, averaging the K values in the window\nthat are closest in intensity to the center pixel.")
    knn_parser.add_argument(
        "input_image", help="Path to the input image file.")
    knn_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    knn_parser.add_argument("--k", type=int, default=6,
                            help="Number of nearest neighbors to consider. Default is 6.")
    knn_parser.add_argument("--window-size", type=int, default=3,
                            help="Size of the square window (e.g., 3 for 3x3). Must be odd. Default is 3.")

    snn_parser = subparsers.add_parser("symmetric-nearest-neighbor", help="Apply Symmetric Nearest Neighbor filter.",
                                       description="Applies a Symmetric Nearest Neighbor filter, selecting the pixel closer in value\nto the center from diametrically opposed pairs within the window, then averages them.")
    snn_parser.add_argument(
        "input_image", help="Path to the input image file.")
    snn_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    snn_parser.add_argument("--window-size", type=int, default=3,
                            help="Size of the square window (e.g., 3 for 3x3). Must be odd. Default is 3.")

    gc_parser = subparsers.add_parser("gamma-correction", help="Apply gamma correction to match image brightness.",
                                      description="Applies gamma correction to the input image to match its mean brightness to an original image.")
    gc_parser.add_argument("input_image", help="Path to the input image file.")
    gc_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    gc_parser.add_argument("--original-image", type=str, required=True,
                           help="Path to the original image file whose mean brightness should be matched.")

    vcd_parser = subparsers.add_parser("van-cittert-deconvolution", help="Apply Van Cittert deconvolution to a blurred image.",
                                       description="Applies the iterative Van Cittert deconvolution algorithm to restore a blurred image.\nRequires a known blurring kernel and number of iterations.")
    vcd_parser.add_argument(
        "input_image", help="Path to the input image file.")
    vcd_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    vcd_parser.add_argument("--kernel", type=str, required=True,
                            help="Convolution kernel as 'R1C1,R1C2;R2C1,R2C2,...'.")
    vcd_parser.add_argument("--iterations", type=int, default=10,
                            help="Number of deconvolution iterations. Default is 10.")

    he_parser = subparsers.add_parser("histogram-equalization", help="Apply standard histogram equalization.",
                                      description="Applies standard histogram equalization to a grayscale image. Saves the equalized image and its histogram plot.")
    he_parser.add_argument("input_image", help="Path to the input image file.")
    he_parser.add_argument(
        "output_image", help="Path to save the equalized output image file.")

    hhe_parser = subparsers.add_parser("hyperbolic-histogram-equalization", help="Apply hyperbolic histogram equalization.",
                                       description="Applies hyperbolic histogram equalization to a grayscale image. Saves the equalized image and its histogram plot.")
    hhe_parser.add_argument(
        "input_image", help="Path to the input image file.")
    hhe_parser.add_argument(
        "output_image", help="Path to save the hyperbolic equalized output image file.")
    hhe_parser.add_argument("--alpha", type=float, default=-1/3,
                            help="Alpha parameter for hyperbolic transformation. Default is -1/3.")

    clut_parser = subparsers.add_parser("contrast-lut", help="Apply a contrast Look-Up Table (LUT).",
                                        description="Applies a pre-defined contrast Look-Up Table to a grayscale image.\nOutput will be a 3-channel image even if input is grayscale.")
    clut_parser.add_argument(
        "input_image", help="Path to the input image file.")
    clut_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")
    clut_parser.add_argument("--lut-type", type=str, default="identity", choices=[
                             "identity", "inverse", "high-contrast", "low-contrast"], help="Type of built-in LUT to apply. Default is 'identity'.")

    custom_clut_parser = subparsers.add_parser("apply-color-lut", help="Apply a specific custom color LUT.",
                                               description="Applies a pre-defined multi-channel color LUT generated by 'gen_lut' to a grayscale image.")
    custom_clut_parser.add_argument(
        "input_image", help="Path to the input image file.")
    custom_clut_parser.add_argument(
        "output_image", help="Path to save the processed output image file.")

    pe_parser = subparsers.add_parser("pointillism", help="Apply a pointillism effect.",
                                      description="Applies a pointillism effect by drawing colored circles at random points.")
    pe_parser.add_argument("input_image", help="Path to the input image file.")
    pe_parser.add_argument(
        "output_image", help="Path to save the pointillized output image file.")
    pe_parser.add_argument("--radius", type=int, default=5,
                           help="Radius of the circles to draw. Default is 5.")

    gc_calc_parser = subparsers.add_parser("global-contrast", help="Calculate global contrast of an image.",
                                           description="Calculates and prints the global contrast (normalized max-min intensity difference) of a grayscale image.")
    gc_calc_parser.add_argument(
        "input_image", help="Path to the input image file.")

    lc_calc_parser = subparsers.add_parser("local-contrast", help="Calculate local contrast of an image.",
                                           description="Calculates and prints the local contrast (mean absolute difference from 8-neighbor average) of a grayscale image.")
    lc_calc_parser.add_argument(
        "input_image", help="Path to the input image file.")

    fourier_parser = subparsers.add_parser("fourier-analysis", help="Perform Fourier Transform analysis and plot results.",
                                           description="Performs Fourier Transform analysis, including magnitude spectrum, low/high-pass reconstructions, and intensity profiles. Saves the analysis as a plot.")
    fourier_parser.add_argument(
        "input_image", help="Path to the input image file.")
    fourier_parser.add_argument("--output-plot-prefix", type=str, default="fourier_analysis",
                                help="Prefix for the output plot filename(s). Default is 'fourier_analysis'.")

    hist_plot_parser = subparsers.add_parser("plot-histogram", help="Plot histogram and CDF of an image.",
                                             description="Plots the histogram and normalized cumulative distribution function (CDF) of a grayscale image. Saves the plot to a file.")
    hist_plot_parser.add_argument(
        "input_image", help="Path to the input image file.")
    hist_plot_parser.add_argument(
        "--title", type=str, default="Image Histogram", help="Title for the histogram plot.")
    hist_plot_parser.add_argument("--output-plot-path", type=str,
                                  default="histogram_plot.png", help="Path to save the histogram plot.")

    profile_plot_parser = subparsers.add_parser("plot-profile", help="Plot image profile from CSV with sampling points.",
                                                description="Plots an image intensity profile from a CSV file (e.g., from ImageJ/Fiji) with marked sampling points.")
    profile_plot_parser.add_argument(
        "csv_file_path", help="Path to the CSV file (must contain 'Distance_(pixels)' and 'Gray_Value' columns).")
    profile_plot_parser.add_argument("--sampling-frequency", type=int, default=50,
                                     help="Frequency in pixels to mark sampling points. Default is 50.")
    profile_plot_parser.add_argument("--output-filename", type=str, default="profile_with_sampling.png",
                                     help="Name of the output plot file. Default is 'profile_with_sampling.png'.")

    zs_thinning_parser = subparsers.add_parser("zhang-suen-thinning", help="Apply Zhang-Suen thinning algorithm.",
                                               description="Applies the Zhang-Suen thinning algorithm to skeletonize a binary image.")
    zs_thinning_parser.add_argument(
        "input_image", help="Path to the input grayscale image (will be binarized internally).")
    zs_thinning_parser.add_argument(
        "output_image", help="Path to save the thinned (skeletonized) output image file.")

    otsu_parser = subparsers.add_parser("otsu-thresholding", help="Apply Otsu's global thresholding.",
                                        description="Applies Otsu's global thresholding to automatically find the optimal binary threshold.")
    otsu_parser.add_argument(
        "input_image", help="Path to the input grayscale image.")
    otsu_parser.add_argument(
        "output_image", help="Path to save the thresholded binary image.")

    itt_parser = subparsers.add_parser("iterative-three-class-thresholding", help="Apply iterative three-class thresholding.",
                                       description="Applies iterative three-class thresholding to classify pixels into dark, middle, and light regions.")
    itt_parser.add_argument(
        "input_image", help="Path to the input grayscale image.")
    itt_parser.add_argument(
        "output_image", help="Path to save the 3-class thresholded image.")

    lot_parser = subparsers.add_parser("local-otsu-thresholding", help="Apply local Otsu's thresholding.",
                                       description="Applies Otsu's thresholding locally, calculating a separate threshold for each small window.")
    lot_parser.add_argument(
        "input_image", help="Path to the input grayscale image.")
    lot_parser.add_argument(
        "output_image", help="Path to save the locally thresholded binary image.")
    lot_parser.add_argument("--window-size", type=int, default=11,
                            help="Size of the square local window (e.g., 11 for 11x11). Must be odd. Default is 11.")

    nn_resize_parser = subparsers.add_parser("resize-nearest-neighbor", help="Resize image using Nearest Neighbor interpolation.",
                                             description="Resizes an image using the Nearest Neighbor interpolation method.")
    nn_resize_parser.add_argument(
        "input_image", help="Path to the input image file.")
    nn_resize_parser.add_argument(
        "output_image", help="Path to save the resized image file.")
    nn_resize_parser.add_argument(
        "--new-width", type=int, required=True, help="Desired new width of the image.")
    nn_resize_parser.add_argument(
        "--new-height", type=int, required=True, help="Desired new height of the image.")

    avg2h_resize_parser = subparsers.add_parser("resize-average-two-horizontal", help="Resize image by averaging two horizontal pixels.",
                                                description="Resizes an image by averaging two horizontal neighbors for downscaling (custom method).")
    avg2h_resize_parser.add_argument(
        "input_image", help="Path to the input image file.")
    avg2h_resize_parser.add_argument(
        "output_image", help="Path to save the resized image file.")
    avg2h_resize_parser.add_argument(
        "--new-width", type=int, required=True, help="Desired new width of the image.")
    avg2h_resize_parser.add_argument(
        "--new-height", type=int, required=True, help="Desired new height of the image.")

    bilinear_resize_parser = subparsers.add_parser(
        "resize-bilinear", help="Resize image using Bilinear interpolation.", description="Resizes an image using Bilinear interpolation.")
    bilinear_resize_parser.add_argument(
        "input_image", help="Path to the input image file.")
    bilinear_resize_parser.add_argument(
        "output_image", help="Path to save the resized image file.")
    bilinear_resize_parser.add_argument(
        "--new-width", type=int, required=True, help="Desired new width of the image.")
    bilinear_resize_parser.add_argument(
        "--new-height", type=int, required=True, help="Desired new height of the image.")

    rgb_min_max_avg_resize_parser = subparsers.add_parser("resize-rgb-min-max-avg", help="Resize RGB image using custom min/max average method.",
                                                          description="Resizes an RGB image by averaging the min and max values of the 4 surrounding pixels for each color channel.")
    rgb_min_max_avg_resize_parser.add_argument(
        "input_image", help="Path to the input image file (must be color).")
    rgb_min_max_avg_resize_parser.add_argument(
        "output_image", help="Path to save the resized image file.")
    rgb_min_max_avg_resize_parser.add_argument(
        "--new-width", type=int, required=True, help="Desired new width of the image.")
    rgb_min_max_avg_resize_parser.add_argument(
        "--new-height", type=int, required=True, help="Desired new height of the image.")

    args = parser.parse_args()

    try:

        if args.command in ["floyd-steinberg", "jarvis-judice-ninke", "ordered-dithering-bayer-8x8"]:
            img = Image.open(args.input_image).convert("L")
            rules = parse_rules(args.rules)
            processed_img = None
            if args.command == "floyd-steinberg":
                processed_img = floyd_steinberg(img, rules)
            elif args.command == "jarvis-judice-ninke":
                processed_img = jarvis_judice_ninke(img, rules)
            elif args.command == "ordered-dithering-bayer-8x8":
                processed_img = ordered_dithering_bayer_8x8(img, rules)
            processed_img.save(args.output_image)
            print(
                f"Image processed successfully and saved to {args.output_image}")

        elif args.command == "ordered-dithering":
            img = Image.open(args.input_image).convert("L")
            if not (args.bayer_n > 0 and (args.bayer_n & (args.bayer_n - 1) == 0)):
                raise ValueError(
                    "Bayer matrix size 'n' must be a power of 2 (e.g., 2, 4, 8, 16).")
            bayer_matrix = generate_bayer_matrix(args.bayer_n)
            processed_img = ordered_dithering(img, bayer_matrix)
            processed_img.save(args.output_image)
            print(
                f"Image processed successfully and saved to {args.output_image}")

        elif args.command == "sine-window":
            sine_window(args.input_image, args.output_image)

        elif args.command == "averaging-filter":
            try:
                w, h = map(int, args.kernel_size.split('x'))
                kernel_size = (w, h)
            except ValueError:
                raise ValueError(
                    "Invalid kernel-size format. Expected 'WxH' (e.g., '3x3').")
            averaging_filter(args.input_image, args.output_image, kernel_size)

        elif args.command == "gamma-correction":
            gamma_correction_to_match_original(
                args.input_image, args.original_image, args.output_image)

        elif args.command == "van-cittert-deconvolution":
            img = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(
                    f"Could not open or find the image at {args.input_image}")
            img_float = img.astype(np.float32) / 255.0

            kernel = parse_kernel(args.kernel)

            deconvolved_image, _ = van_cittert_deconvolution(
                img_float, kernel, args.iterations)
            deconvolved_image_uint8 = np.clip(
                deconvolved_image * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(args.output_image, deconvolved_image_uint8)
            print(
                f"Image deconvolved successfully and saved to {args.output_image}")

        elif args.command == "histogram-equalization":
            histogram_equalization(args.input_image, args.output_image)

        elif args.command == "hyperbolic-histogram-equalization":
            hyperbolic_histogram_equalization(
                args.input_image, args.output_image, args.alpha)

        elif args.command == "k-trimmed-mean":
            k_trimmed_mean_filter(
                args.input_image, args.output_image, args.k, args.window_size)

        elif args.command == "k-nearest-neighbor":
            k_nearest_neighbor_filter(
                args.input_image, args.output_image, args.k, args.window_size)

        elif args.command == "symmetric-nearest-neighbor":
            symmetric_nearest_neighbor_filter(
                args.input_image, args.output_image, args.window_size)

        elif args.command == "zhang-suen-thinning":
            zhang_suen_thinning(args.input_image, args.output_image)

        elif args.command == "otsu-thresholding":
            otsu_thresholding(args.input_image, args.output_image)

        elif args.command == "iterative-three-class-thresholding":
            iterative_three_class_thresholding(
                args.input_image, args.output_image)

        elif args.command == "local-otsu-thresholding":
            local_otsu_thresholding(
                args.input_image, args.output_image, args.window_size)

        elif args.command == "pointillism":
            pointillism_effect(
                args.input_image, args.output_image, args.radius)

        elif args.command == "resize-nearest-neighbor":
            resize_nearest_neighbor(
                args.input_image, args.output_image, args.new_width, args.new_height)

        elif args.command == "resize-average-two-horizontal":
            resize_average_two_horizontal(
                args.input_image, args.output_image, args.new_width, args.new_height)

        elif args.command == "resize-bilinear":
            resize_bilinear(args.input_image, args.output_image,
                            args.new_width, args.new_height)

        elif args.command == "resize-rgb-min-max-avg":
            resize_rgb_min_max_avg(
                args.input_image, args.output_image, args.new_width, args.new_height)

        elif args.command == "contrast-lut":
            lut_rgb = generate_basic_lut(args.lut_type)
            contrast_lut(args.input_image, args.output_image, lut_rgb)

        elif args.command == "apply-color-lut":

            lut_r, lut_g, lut_b = gen_lut()
            custom_lut_rgb = [lut_b, lut_g, lut_r]
            contrast_lut(args.input_image, args.output_image, custom_lut_rgb)

        elif args.command == "global-contrast":
            contrast = calculate_global_contrast(args.input_image)
            if contrast is not None:
                print(
                    f"Global contrast of '{args.input_image}': {contrast:.4f}")

        elif args.command == "local-contrast":
            contrast = calculate_local_contrast(args.input_image)
            if contrast is not None:
                print(
                    f"Local contrast of '{args.input_image}': {contrast:.4f}")

        elif args.command == "fourier-analysis":
            process_image_fourier(args.input_image, args.output_plot_prefix)

        elif args.command == "plot-histogram":
            img = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(
                    f"Could not open or find the image at {args.input_image}")
            plot_histogram(img, args.title, args.output_plot_path)

        elif args.command == "plot-profile":
            plot_profile_with_sampling(
                args.csv_file_path, args.sampling_frequency, args.output_filename)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
