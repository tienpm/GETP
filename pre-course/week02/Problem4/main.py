import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from utils import save_np, plot_images

def load_image(file_path):
    image = np.load(file_path)
    return image

def apply_convolution(images, kernel, stride=1, padding=0):
    '''
    Implement 2D Convolution Using Numpy
    =================================================================================================
    Arguments:
        + images: np.array of shape (B, input_H, input_W)
        + kernel: np.array of shape (kernel_H, kernel_W)
        + stride: integer
        + padding: integer
    Outputs:
        + output_images: np.array of shape (B, input_H, input_W)
    '''
    ### TODO: fill in here ###
    m, n_H_init, n_W_init = images.shape
    kernel_H, kernel_W = kernel.shape
    # Compute the dimensions of the CONV output volume
    n_H = int((n_H_init - kernel_H + (2 * padding)) / stride + 1)
    n_W = int((n_W_init - kernel_W + (2 * padding)) / stride + 1)

    # Initialize the output volume with zeros.
    Y = np.zeros((m, n_H, n_W))

    # Create X by padding input images
    X = np.pad(images, ((0,), (padding,), (padding,)), 'constant', constant_values=0)

    for i in range(m):  # loop over the batch of input images
        X_prev_pad = X[i]
        for h in range(n_H):  # loop over vertical axis of the output volume
            vert_start = h * stride
            vert_end = h * stride + kernel_H
            for w in range(n_W):  # loop over horizontal axis of the output volume    
                horiz_start = w * stride
                horiz_end = w * stride + kernel_W

                # Use the corners to define the (2D) slice of a_prev_pad
                x_slice_prev = X_prev_pad[vert_start:vert_end, horiz_start:horiz_end]

                # Convolve the (2D) slice
                s = np.multiply(x_slice_prev, kernel)
                Y[i, h, w] = np.sum(s)

    return Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Detection using Sobel Filter')
    parser.add_argument('--number_imgs', default=8, type=int,
                        help='The number images you want to save')
    parser.add_argument('--filename', default="sobel", type=str,
                        help='A file name of images to save')
    args = parser.parse_args()

    input_file_path = "input_image.npy"

    input_images = load_image(input_file_path)
    num_images = input_images.shape[0]


    # Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    result_x = apply_convolution(input_images, sobel_x, stride=1, padding=1)
    result_y = apply_convolution(input_images, sobel_y, stride=1, padding=1)

    # Combine the results
    result = np.sqrt(result_x**2 + result_y**2)

    # Random 8 images to save
    img_order = np.random.choice(num_images, args.number_imgs, replace=False)
    imgs = []
    for i in img_order:
        imgs.append(input_images[i])
        imgs.append(result[i])

    save_np(imgs, file_name=f"{args.filename}.npy")
    plot_images(imgs, file_name=f"{args.filename}.png")
    # plot_images(imgs)
    '''
    =================================================================================================
    Save and submit a portion of the processed 32 images. 
    You are free to choose any number of images (recommend 4~8)."
    '''


