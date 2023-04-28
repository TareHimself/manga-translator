import cv2
import numpy as np
from PIL import Image


def find_largest_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return rect, box


def main():
    contour = np.array(
        [[[100, 100]], [[200, 150]], [[300, 200]], [[200, 300]], [[100, 300]]],
        dtype=np.int32,
    )

    largest_rect, box = find_largest_rectangle(contour)

    img = np.zeros((400, 400, 3), dtype=np.uint8)

    done = Image.fromarray(cv2.drawContours(img.copy(), [contour], 0, (0, 255, 0), 2))
    done.show()

    done = Image.fromarray(cv2.drawContours(img.copy(), [box], 0, (255, 0, 0), 2))
    done.show()


main()
