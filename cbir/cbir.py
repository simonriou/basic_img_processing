import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from typing import Tuple

def extract_features() -> Tuple[np.ndarray, np.ndarray]:
    """Resturns the Gabor features matrix and the Lum and Chrom values matrix"""
    # Open dataset as read
    filename = r'CBIR/dataset.csv'
    with open(filename, "r") as file:
        data = np.array(list(csv.reader(file, delimiter=";")))

    # Gabor Feature matrix
    X_str = data[:, 3:27].astype(float)
    # Lum and Chrom values matrix
    Y_str = data[:, 27:30].astype(float)

    return X_str, Y_str

def get_lum_chrom(image: str, dataset: np.ndarray) -> Tuple[float, float, float]:
    """Returns the Lum, Cr and Cb values of the image"""
    # Get the number of the image
    num = int(image.split('.')[0])
    # Get the row of the image in the Y of the dataset
    row = dataset[num]
    # Get the Lum and Chrom values
    lum, cr, cb = row

    return lum, cr, cb

def get_sim_imges_lum(nb: int, n: int, dataset: np.ndarray, lum: float, cr: float, cb: float) -> list:
    """Returns the nb most similar images to the image n (excluding n) using the Lum and Chrom values"""
    # Find the nb most similar images to the image n (excluding n)
    # The similarity is based on the Lum and Chrom values
    mainclass_n, subclass_n = get_class(n)
    distances = np.abs(dataset[:, 0] - lum) + np.abs(dataset[:, 1] - cr) + np.abs(dataset[:, 2] - cb)
    sorted_sim_images = sorted([(i, float(dist)) for i, dist in enumerate(distances) if i != n], key=lambda x: x[1])
    final_imgs = []
    for i in range(len(sorted_sim_images)):
        if(len(final_imgs) == nb):
            break
        if (mainclass_n, subclass_n) == get_class(sorted_sim_images[i][0]):
            final_imgs.append(sorted_sim_images[i])

    return final_imgs

def get_sim_imges_gabor(nb: int, n: int, dataset: np.ndarray, gabor: np.ndarray) -> list:
    """Returns the nb most similar images to the image n (excluding n) using the Gabor values"""
    # gabor is a list of 24 values
    # Find the nb most similar images to the image n (excluding n)
    # The similarity is based on the Gabor values
    mainclass_n, subclass_n = get_class(n)
    distances = np.linalg.norm(dataset - gabor, axis=1)
    sorted_sim_images = sorted([(i, float(dist)) for i, dist in enumerate(distances) if i != n], key=lambda x: x[1])
    final_imgs = []
    for i in range(len(sorted_sim_images)):
        if(len(final_imgs) == nb):
            break
        if (mainclass_n, subclass_n) == get_class(sorted_sim_images[i][0]):
            final_imgs.append(sorted_sim_images[i])

    return final_imgs

def get_class(image_nb: int) -> int:
    """Returns the class of the image"""
    # Open the dataset as read
    filename = r'CBIR/dataset.csv'
    with open(filename, "r") as file:
        data = np.array(list(csv.reader(file, delimiter=";")))

    # The class and subclass of the image are the second and third columns
    # Get the row of the image in the dataset
    row = data[image_nb]
    # Get the class of the image
    mainclass = int(row[1])
    subclass = int(row[2])

    return mainclass, subclass

def main():
    mode = int(input("Enter mode (0: Lum and Chrom, 1: Gabor, 99: Debug): "))
    image_number = int(input("Enter the image number: "))
    print(f"Image: {image_number}.jpg | Class: {get_class(image_number)}")

    if mode == 0:
        # Extract the features from the dataset (using lum, cr and cb)
        _, Y = extract_features()
        lum, cr, cb = get_lum_chrom(f'{image_number}.jpg', Y)

        sim_images = get_sim_imges_lum(5, image_number, Y, lum, cr, cb)

        # Display the 5 most similar images (get them from cbir/patches)
        for im in sim_images:
            plt.subplot(1, 6, sim_images.index(im) + 1)
            plt.axis('off')
            plt.imshow(plt.imread(f'CBIR/patches/{im[0]}.jpg'))
            plt.subplots_adjust(wspace=0.5)
            plt.title(f"L: {Y[im[0]][0]:.2f}\nCr: {Y[im[0]][1]:.2f}\nCb: {Y[im[0]][2]:.2f}")

        plt.subplot(1, 6, 6)
        plt.imshow(plt.imread(f'CBIR/patches/{image_number}.jpg'))
        plt.subplots_adjust(wspace=0.5)
        plt.title(f"L: {lum:.2f}\nCr: {cr:.2f}\nCb: {cb:.2f}")
        plt.axis('off')
        plt.show()
    elif mode == 99:
        # Debug mode
        print(get_class(image_number))
    else:
        # Extract the features from the dataset (using the gabor features)
        X, _ = extract_features()
        gabor = X[image_number]

        sim_images = get_sim_imges_gabor(5, image_number, X, gabor)

        # Display the 5 most similar images (get them from cbir/patches)
        for im in sim_images:
            plt.subplot(1, 6, sim_images.index(im) + 1)
            plt.axis('off')
            plt.imshow(plt.imread(f'CBIR/patches/{im[0]}.jpg'))
            plt.subplots_adjust(wspace=0.5)
            plt.title(f"{im[1]:.2f}")

        plt.subplot(1, 6, 6)
        plt.imshow(plt.imread(f'CBIR/patches/{image_number}.jpg'))
        plt.subplots_adjust(wspace=0.5)
        plt.title(f"{np.linalg.norm(gabor):.2f}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()
