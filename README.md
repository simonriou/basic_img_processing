# Image Mosaicking and Sketching Project

This project consists of two scripts: an image **mosaicking** script and an image **sketching** script. The mosaicking script creates a mosaic image from a target image by matching blocks in the target with patches from a folder. The sketching script generates a pencil sketch effect from a given image by inverting the grayscale image and applying a Gaussian blur.

## Project Structure

- Mosaicking Script: Generates a mosaic representation of an image.
- Sketching Script: Converts an image into a pencil sketch using Gaussian blur.

## Requirements

Install the required Python libraries using:
```bash
pip install numpy matplotlib scikit-image scipy
```

## Mosaicking Script
### Usage
- Place your target image in the `Target_images/` directory (or use one of the original ones) and adjust the name of the target file on line 21
- Place your patches in the `patches` directory (or use the original ones)
- Run the script in your Python environment
