import sys

from random.contrast import *


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 random.py <PATH_IN> <VARIANT> <PATH_OUT>")
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
