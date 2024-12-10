import skimage
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import concurrent.futures
from tqdm import tqdm

### ------------- INITIALISATION ------------- ###

# Parameters
block_size = 16 # Size of one block
patch_dir = 'patches' # Directory where patches are stored
multi_threading = True # Set this to True to use multiple cores of your processor for computations

# Function for loading patches using multi-threading
def load_and_resize_patch(filename):
    """Loads and saves patches"""
    img = skimage.io.imread(os.path.join(patch_dir, filename))
    mean_color = np.mean(img, axis=(0, 1))
    return img, mean_color

### ------------- PATCH LOADING ------------- ###

patchesStart = time.time()
patches = []

if multi_threading:
    # Load patches using multi-threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_and_resize_patch, os.listdir(patch_dir))
        patches.extend(results)
else:
    # W/out multi-threading
    for filename in os.listdir(patch_dir):
        img = skimage.io.imread(os.path.join(patch_dir, filename))
        mean_color = np.mean(img, axis=(0, 1))
        patches.append((img, mean_color))

patchesEnd = time.time()
print(f"Patches loading time : {patchesEnd - patchesStart} seconds" )

### ------------- IMAGE RESIZE ------------- ###

# Making sure the target image contains a whole number of blocks
target_img = skimage.io.imread('./Target_images/NewYork.jpg')
target_img_resized = skimage.transform.resize(target_img,
                                              (target_img.shape[0] // block_size*block_size,
                                               target_img.shape[1] // block_size*block_size),
                                               anti_aliasing=True)

### ------------- BLOCKS ------------- ###

# Divide target image in blocks, calc mean
blocks = []
for i in range(0, target_img_resized.shape[0], block_size):
    for j in range(0, target_img_resized.shape[1], block_size):
        block = target_img_resized[i:i+block_size, j:j+block_size]
        mean_col = np.mean(block, axis=(0, 1))
        blocks.append((block, mean_col, (i, j))) # Save block, mean color and location of the block

### ------------- MEAN COLORS ------------- ###

# Color approx. of the target image
# color_approx = np.zeros_like(target_img_resized)
# for _, mean_col, (i, j) in blocks:
#     color_approx[i:i+block_size, j:j+block_size] = mean_col

# plt.imshow(color_approx)
# plt.axis('off')
# plt.show()

### ------------- MOSAICKING ------------- ###

# Function for chosing the best patch (mutli-threading)
def mosaicking(mean_col, patch_means, patches, block_size, i, j):
    """Processes a single block for the mosaic."""
    dists = np.linalg.norm(patch_means - mean_col * 255, axis=1)
    best_index = np.argmin(dists)
    best_patch = patches[best_index][0]
    best_patch = skimage.transform.resize(best_patch, (block_size, block_size), anti_aliasing=True)
    return i, j, best_patch

mosaicStart = time.time()

# Convert mean colors to a numpy array for vectorized distance calculations
patch_means = np.array([patch_mean for _, patch_mean in patches])

# For each block, calculate distances to all patches in one go
mosaic_img = np.zeros_like(target_img_resized)

if multi_threading:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _, mean_col, (i, j) in blocks: # For each block
            # Add the best patch to the futures list
            futures.append(executor.submit(mosaicking, mean_col, patch_means, patches, block_size, i, j))
        
        # Then for each patch
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Building Mosaic", unit="block"):
            i, j, best_patch = future.result()
            mosaic_img[i:i+block_size, j:j+block_size] = best_patch # Add it to the mosaic
else:
    for _, mean_col, (i, j) in tqdm(blocks, desc="Building Mosaic", unit="block"): # Progress bar
        # Vectorized calculation of distances
        dists = np.linalg.norm(patch_means - mean_col * 255, axis=1)
        best_index = np.argmin(dists)
        best_patch = patches[best_index][0]
        best_patch = skimage.transform.resize(best_patch, (block_size, block_size), anti_aliasing=True)
        mosaic_img[i:i+block_size, j:j+block_size] = best_patch

mosaicEnd = time.time()
print(f"Elapsed time for mosaicking : {mosaicEnd - mosaicStart} seconds.")

plt.imshow(mosaic_img)
plt.axis('off')
plt.show()
