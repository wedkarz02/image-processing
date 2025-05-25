import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def plot_histogram(image_data, title, output_plot_path):
    hist = cv2.calcHist([image_data], [0], None, [256], [0, 256]).flatten()
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max() if cdf.max() > 0 else cdf

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel('Poziom jasności piksela')
    plt.ylabel('Liczba pikseli / Skumulowana liczba pikseli (Znormalizowana)')

    plt.plot(hist, color='blue', label='Histogram')
    plt.fill_between(range(256), hist, color='lightblue', alpha=0.5)

    plt.plot(cdf_normalized, color='red',
             label='Skumulowany Histogram (Znormalizowany)')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, 255])
    plt.tight_layout()

    plt.savefig(output_plot_path)
    print(f"plot '{title}' saved as: {output_plot_path}")
    plt.close()


def calculate_and_plot_original_histograms(image_path, output_plot_path=None):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(
                f"err: image not found: {image_path}")
            return

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten()

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max() if cdf.max() > 0 else cdf

        plt.figure(figsize=(10, 6))
        plt.title('Histogram i Skumulowany Histogram')
        plt.xlabel('Poziom jasności piksela')
        plt.ylabel('Liczba pikseli / Skumulowana liczba pikseli (znormalizowana)')

        plt.plot(hist, color='blue', label='Histogram')
        plt.fill_between(range(256), hist, color='lightblue', alpha=0.5)

        plt.plot(cdf_normalized, color='red',
                 label='Skumulowany Histogram (znormalizowany)')

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0, 255])
        plt.tight_layout()

        if output_plot_path:
            plt.savefig(output_plot_path)
            print(f"plot saved as: {output_plot_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"err: {e}")


def histogram_equalization(image_path, output_image_path):
    try:
        if not os.path.exists(image_path):
            print(f"err: file not found: {image_path}")
            return

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"err: image not found or invalid format: {image_path}")
            return

        equalized_img = cv2.equalizeHist(img)

        cv2.imwrite(output_image_path, equalized_img)
        print(f"equalized img saved as: {output_image_path}")

        output_dir = os.path.dirname(output_image_path)
        output_filename_base = os.path.splitext(
            os.path.basename(output_image_path))[0]
        output_hist_plot_path = os.path.join(
            output_dir, f"{output_filename_base}_hist.png")
        plot_histogram(
            equalized_img, 'Histogram i Skumulowany Histogram Wyrównanego Obrazu', output_hist_plot_path)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        cdf = hist.cumsum()

        cdf_m = np.ma.masked_equal(cdf, 0)
        if cdf_m.count() > 0:
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
        else:
            cdf_final = np.zeros_like(cdf, dtype='uint8')

        g_values = [10, 15, 20]

        print("\nWartości szarości w obrazie wyjściowym H_equal(g) dla podanych g (przed zaokrągleniem przez cv2):")
        for g in g_values:
            if g >= 0 and g <= 255:
                h_equal_g = cdf_final[g]
                print(f"  dla g = {g}: H_equal(g) = {h_equal_g}")
            else:
                print(f"  Wartość g={g} jest poza zakresem [0, 255].")

    except Exception as e:
        print(f"err: {e}")


def hyperbolic_histogram_equalization(image_path, output_image_path, alpha=-1/3):
    try:
        if not os.path.exists(image_path):
            print(f"err: file not found: {image_path}")
            return

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"err: image not found or invalid format: {image_path}")
            return

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
            f"hiperbolized img (alpha={alpha}) saved as: {output_image_path}")

        output_dir = os.path.dirname(output_image_path)
        output_filename_base = os.path.splitext(
            os.path.basename(output_image_path))[0]
        output_hist_plot_path = os.path.join(
            output_dir, f"{output_filename_base}_hist.png")
        plot_histogram(hyper_equalized_img,
                       f'Histogram i Skumulowany Histogram Hiperbolizowanego Obrazu (alpha={alpha})', output_hist_plot_path)

        g_values = [10, 15, 20]

        print(
            f"\nWartości szarości w obrazie wyjściowym H_hyper(g) dla podanych g (alpha={alpha}):")
        for g in g_values:
            if g >= 0 and g <= 255:
                h_hyper_g = map_function[g]
                print(f"  dla g = {g}: H_hyper(g) = {h_hyper_g}")
            else:
                print(f"  Wartość g={g} jest poza zakresem [0, 255].")

    except Exception as e:
        print(f"err: {e}")


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 hist.py <PATH_IN> <VARIANT> <PATH_OUT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]
    path_out = sys.argv[3]

    if var == "a":
        calculate_and_plot_original_histograms(path_in, path_out)
    elif var == "b":
        histogram_equalization(path_in, path_out)
    elif var == "c":
        hyperbolic_histogram_equalization(path_in, path_out)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
