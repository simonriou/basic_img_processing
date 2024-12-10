import skimage
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = skimage.io.imread('input.jpg')

# Convert to grayscale
gray_img = skimage.color.rgb2gray(img)

# Image inversion
inv_img = 255 - gray_img

# Define function for Gaussian Blur & final sketch (adjust parameters)
def compute(inverted_img, sigma):
    # Gaussian blur
    blurred_img = skimage.filters.gaussian(inv_img, sigma)

    sketch = (blurred_img*255)/inv_img
    sketch = np.clip(sketch, 0, 255)

    imgs = [gray_img, inv_img, blurred_img, sketch]
    rows, cols = 2, 2
    fig, axs = plt.subplots(rows, cols, figsize=(10, 7))

    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i], cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return sketch

sigma = 16
sketch = compute(inv_img, sigma)
