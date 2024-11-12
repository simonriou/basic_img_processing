import skimage
import skimage.transform
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os

# Parameters
block_size = 50
patch_dir = 'patches'
target_dir = 'Target_images'

# Load patches
patches = []
for filename in os.listdir(patch_dir):
    img = skimage.io.imread(os.path.join(patch_dir, filename))
    mean_color = np.mean(img, axis=(0, 1))
    patches.append((img, mean_color))

# Making sure the target image contains a whole number of blocks
target_img = skimage.io.imread('./Target_images/Rainbow.jpg')
target_img_resized = skimage.transform.resize(target_img,
                                              (target_img.shape[0] // block_size*block_size,
                                               target_img.shape[1] // block_size*block_size),
                                               anti_aliasing=True)

# Divide target image in blocks, calc mean
blocks = []
for i in range(0, target_img_resized.shape[0], block_size):
    for j in range(0, target_img_resized.shape[1], block_size):
        block = target_img_resized[i:i+block_size, j:j+block_size]
        mean_col = np.mean(block, axis=(0, 1))
        blocks.append((block, mean_col, (i, j))) # Save block, mean color and location of the block

# Color approx. of the target image
color_approx = np.zeros_like(target_img_resized)
for _, mean_col, (i, j) in blocks:
    color_approx[i:i+block_size, j:j+block_size] = mean_col

# Patch comparaison
mosaic_img = np.zeros_like(target_img_resized)
for _, mean_col, (i, j) in blocks:
    dist = []
    for _, patch_mean in patches:
        dist.append(distance.euclidean(mean_col * 255, patch_mean))
    best_index = np.argmin(dist)
    best_patch = patches[best_index][0]
    best_patch = skimage.transform.resize(best_patch, (block_size, block_size), anti_aliasing=True)
    mosaic_img[i:i+block_size, j:j+block_size] = best_patch

plt.imshow(mosaic_img)
plt.axis('off')
plt.show()