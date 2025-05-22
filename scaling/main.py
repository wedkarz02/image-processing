import sys
from PIL import Image
import numpy as np


def resize_nearest_neighbor(image, new_width, new_height):
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

    return Image.fromarray(result)


def resize_average_two_horizontal(image, new_width, new_height):
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
            avg_pixel = ((pixel1.astype(np.uint16) + pixel2.astype(np.uint16)) // 2).astype(np.uint8)

            result[y_new, x_new] = avg_pixel

    return Image.fromarray(result)


def main():
    if len(sys.argv) != 6:
        print("USAGE: python3 main.py <PATH_IN> <WIDTH> <HEIGHT> <ALG> <PATH_OUT> ")
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
    else:
        sys.exit(2)

    resized.save(path_out)
    print(f"Saved image at: {path_out}")


if __name__ == "__main__":
    main()
