import sys
import cv2 as cv
import numpy as np

def pixel_transform(img, f):
    for row in img:
        for pixel in row:
            pixel = f(pixel)
    return img       

def rotate_rgb(img, k):
    def rotate_rgb_helper(pixel):
        pixel[0], pixel[1], pixel[2] = pixel[k%3], pixel[(k+1)%3], pixel[(k+2)%3]
        return pixel

    return pixel_transform(img, rotate_rgb_helper)

def simple_grayscale(img):
    def simple_grayscale_helper(pixel):
        gray = (pixel[0] + pixel[1] + pixel[2])/3
        pixel[0] = gray
        pixel[1] = gray
        pixel[2] = gray
    return pixel_transform(img, simple_grayscale_helper)

def weighted_grayscale(img, weights):
    def weighted_grayscale_helper(pixel):
        gray = weights[0]*pixel[0] + weights[1]*pixel[1] + weights[2]*pixel[2]
        pixel[0] = gray
        pixel[1] = gray
        pixel[2] = gray
    return pixel_transform(img, weighted_grayscale_helper)

def main():
    img = cv.imread(sys.argv[1])
    cv.imshow('original', img)
    # cv.imshow('rotate_rgb', rotate_rgb(img, 1))
    # cv.imshow('simple_grayscale', simple_grayscale(img))
    # cv.imshow('weighted_grayscale', weighted_grayscale(img, [0.3, 0.59, 0.11]))
    edgeDetectionKernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    print(edgeDetectionKernel)
    cv.imshow('cv', cv.filter2D(weighted_grayscale(img, [0.3, 0.59, 0.11]), -1, edgeDetectionKernel))
    cv.waitKey(0)
    cv.destroyAllWindows()

main()       