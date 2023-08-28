import cv2
import numpy as np


def apply_affine(img):
    rows, cols = img.shape[:2]
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.3*(cols-1)),rows-1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
    cv2.imwrite("target/affine.png",img_output)

def apply_rotation(img):
    img_output = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite("target/rotate180.png",img_output)


img = cv2.imread('target/Pepsi-vs-Coca-Cola.jpeg')
apply_affine(img.copy())
apply_rotation(img.copy())
apply_grayscale(img.copy())