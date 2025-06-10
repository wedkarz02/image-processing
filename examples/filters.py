import sys

from filters import *


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 filters.py <PATH_IN> <VARIANT> <PATH_OUT>")
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
    elif var == "vc":
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ]) / 256.0
        iters = 2
        try:
            img_blurred_bgr = cv2.imread(
                path_in, cv2.IMREAD_UNCHANGED)

            if img_blurred_bgr is None:
                raise FileNotFoundError(
                    f"failed to load image")

            if img_blurred_bgr.dtype == np.uint8:
                img_blurred_normalized = img_blurred_bgr.astype(float) / 255.0
            elif img_blurred_bgr.dtype == np.uint16:
                img_blurred_normalized = img_blurred_bgr.astype(
                    float) / 65535.0
            else:
                img_blurred_normalized = img_blurred_bgr.astype(float)

            if img_blurred_normalized.ndim == 3:
                img_blurred_gray = cv2.cvtColor(
                    img_blurred_normalized, cv2.COLOR_BGR2GRAY)
            else:
                img_blurred_gray = img_blurred_normalized

            deconvolved_image, difference_images = van_cittert_deconvolution(
                img_blurred_gray, kernel, iters
            )

            cv2.imwrite("output/vc/deconvolved_image.png",
                        (deconvolved_image * 255).astype(np.uint8))
            print(f"deconvuled img saved as: output/vc/deconvolved_image.png")

            for i, diff_img in enumerate(difference_images):
                max_diff = diff_img.max()
                if max_diff > 0:
                    normalized_diff_img = (
                        diff_img / max_diff * 255).astype(np.uint8)
                else:
                    normalized_diff_img = np.zeros_like(
                        diff_img, dtype=np.uint8)

                cv2.imwrite(
                    f"output/vc/difference_iteration_{i+1}.png", normalized_diff_img)
                print(
                    f"diff img (iter {i+1}) saved as: output/vc/difference_iteration_{i+1}.png")

        except Exception as e:
            print(f"err: {e}")
    elif var == "ktm":
        filtered_image = k_trimmed_mean_filter(path_in)
        cv2.imwrite(path_out, filtered_image)
        print(f"img saved as: {path_out}")
    elif var == "knn":
        filtered_image = k_nearest_neighbor_filter(path_in)
        cv2.imwrite(path_out, filtered_image)
        print(f"img saved as: {path_out}")
    elif var == "snn":
        filtered_image = symmetric_nearest_neighbor_filter(path_in)
        cv2.imwrite(path_out, filtered_image)
        print(f"img saved as: {path_out}")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
