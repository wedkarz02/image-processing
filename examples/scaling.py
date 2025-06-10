import sys

from scaling import *


def main():
    if len(sys.argv) != 6:
        print("USAGE: python3 scaling.py <PATH_IN> <WIDTH> <HEIGHT> <ALG> <PATH_OUT> ")
        sys.exit(1)

    path_in = sys.argv[1]
    width_out = int(sys.argv[2])
    height_out = int(sys.argv[3])
    algorithm = sys.argv[4].lower()
    path_out = sys.argv[5]

    img = Image.open(path_in).convert("RGB")

    if algorithm == "a":
        resized = resize_nearest_neighbor(img, width_out, height_out)
    elif algorithm == "b":
        resized = resize_average_two_horizontal(img, width_out, height_out)
    elif algorithm == "c":
        resized = resize_bilinear(img, width_out, height_out)
    elif algorithm == "d":
        resized = resize_rgb_min_max_avg(img, width_out, height_out)
    else:
        sys.exit(2)

    resized.save(path_out)
    print(f"Saved image at: {path_out}")


if __name__ == "__main__":
    main()
