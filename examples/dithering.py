import sys
from PIL import Image

from dithering import *


def main():
    if len(sys.argv) != 5:
        print("USAGE: python3 dithering.py <PATH_IN> <VARIANT> <ALG> <PATH_OUT> ")
        sys.exit(1)

    path_in = sys.argv[1]
    variant = sys.argv[2].lower()
    algorithm = sys.argv[3].lower()
    path_out = sys.argv[4]

    img = Image.open(path_in).convert("L")

    rules = []
    if variant == "a":
        rules = [
            (128, 0),
            (256, 255),
        ]
    elif variant == "b":
        rules = [
            (20, 0),
            (40, 64),
            (60, 128),
            (120, 192),
            (256, 255)
        ]
    elif variant == "c":
        rules = [
            (75, 50),
            (125, 100),
            (175, 150),
            (256, 200)
        ]
    else:
        print("Error: Invalid variant. Supported: 'a', 'b', 'c'.")
        sys.exit(1)

    if algorithm == "fs":
        dithered_img = floyd_steinberg(img, rules)
    elif algorithm == "jjn":
        dithered_img = jarvis_judice_ninke(img, rules)
    elif algorithm == "ordered":
        dithering_matrix = np.array([
            [6, 14, 2, 8],
            [4, 0, 10, 11],
            [12, 15, 5, 1],
            [9, 3, 13, 7]
        ], dtype=np.float32)
        dithered_img = ordered_dithering(img, dithering_matrix)
    elif algorithm == "bayer":
        dithered_img = ordered_dithering_bayer_8x8(img, rules)
    else:
        sys.exit(2)

    dithered_img.save(path_out)
    print(f"Saved image at: {path_out}")


if __name__ == "__main__":
    main()
