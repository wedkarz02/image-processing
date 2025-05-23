import sys
import random
from PIL import Image, ImageDraw


def pointillism_effect(image, radius):
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

    return output


def main():
    if len(sys.argv) != 4:
        print("USAGE: python3 pointillism.py <PATH_IN> <RADIUS> <PATH_OUT> ")
        sys.exit(1)

    path_in = sys.argv[1]
    radius = int(sys.argv[2])
    path_out = sys.argv[3]

    image = Image.open(path_in).convert("RGB")
    result = pointillism_effect(image, radius)
    result.save(path_out)
    print(f"Saved image at: {path_out}")


if __name__ == "__main__":
    main()
