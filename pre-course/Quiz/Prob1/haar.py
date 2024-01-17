import argparse

import cv2
import numpy as np


def normalize(img):
    new_img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return new_img


def haar_wavelet_transform(img, depth=1):
    """
    IMPLEMENT THIS FUNCTION
    """
    if depth == 0:
        return img

    new_img = np.copy(img)

    m, n = new_img.shape
    L = np.zeros((m // 2, n))
    H = np.zeros((m // 2, n))
    for i in range(m // 2):
        L[i, :] = (new_img[2 * i, :] + new_img[2 * i + 1, :]) / np.sqrt(2)
        H[i, :] = (new_img[2 * i, :] - new_img[2 * i + 1, :]) / np.sqrt(2)

    LL = np.zeros((m // 2, n // 2))
    LH = np.zeros((m // 2, n // 2))
    HL = np.zeros((m // 2, n // 2))
    HH = np.zeros((m // 2, n // 2))
    for j in range(n // 2):
        LL[:, j] = (L[:, 2 * j] + L[:, 2 * j + 1]) / np.sqrt(2)
        LH[:, j] = (L[:, 2 * j] - L[:, 2 * j + 1]) / np.sqrt(2)
        HL[:, j] = (H[:, 2 * j] + H[:, 2 * j + 1]) / np.sqrt(2)
        HH[:, j] = (H[:, 2 * j] - H[:, 2 * j + 1]) / np.sqrt(2)

    new_img = np.concatenate(
        (np.concatenate((LL, LH), axis=1), np.concatenate((HL, HH), axis=1)), axis=0
    )
    new_img = haar_wavelet_transform(new_img, depth - 1)

    return new_img


def main(args):
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    transformed_image = haar_wavelet_transform(image, depth=args.depth)
    cv2.imwrite(args.output, transformed_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, required=True, help="output image path")
    parser.add_argument(
        "--depth", type=int, required=True, help="depth of Haar wavelet transform"
    )
    args = parser.parse_args()
    main(args)

