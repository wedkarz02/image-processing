
import numpy as np
import cv2
import sys


def zhang_suen_thinning(image_path):
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
                    if img[r, c]:
                        p = [0] * 10
                        p[2] = int(img[r, c+1])
                        p[3] = int(img[r+1, c+1])
                        p[4] = int(img[r+1, c])
                        p[5] = int(img[r+1, c-1])
                        p[6] = int(img[r, c-1])
                        p[7] = int(img[r-1, c-1])
                        p[8] = int(img[r-1, c])
                        p[9] = int(img[r-1, c+1])

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
                img[r, c] = 0
                changed = True

            pixels_to_delete_step2 = []
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if img[r, c]:
                        p = [0] * 10
                        p[2] = int(img[r, c+1])
                        p[3] = int(img[r+1, c+1])
                        p[4] = int(img[r+1, c])
                        p[5] = int(img[r+1, c-1])
                        p[6] = int(img[r, c-1])
                        p[7] = int(img[r-1, c-1])
                        p[8] = int(img[r-1, c])
                        p[9] = int(img[r-1, c+1])

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
                img[r, c] = 0
                changed = True

        skeleton_01_no_pad = img[1:-1, 1:-1]
        skeleton_255_output = (1 - skeleton_01_no_pad) * 255
        return skeleton_255_output.astype(np.uint8)

    except FileNotFoundError as e:
        print(f"err: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"err: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 thinning.py <PATH_IN> <VARIANT> <PATH_OUT>")
        sys.exit(1)

    path_in = sys.argv[1]
    var = sys.argv[2]
    path_out = sys.argv[3]

    if var == "a":
        result = zhang_suen_thinning(path_in)
        cv2.imwrite(path_out, result)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
