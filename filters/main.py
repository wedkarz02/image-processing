import sys
import cv2
import numpy as np


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


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 main.py <PATH_IN> <VARIANT> <PATH_OUT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]
    path_out = sys.argv[3]

    if var == "a":
        sine_window(path_in, path_out)
    elif var == "b":
        averaging_filter(path_in, path_out)
    elif var == "c":
        gamma_correction_to_match_original(
            path_in, "../../obrazy/ptaki.png", path_out)
    elif var == "d":
        averaging_filter(path_in, path_out)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
