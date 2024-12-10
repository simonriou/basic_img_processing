import skimage
import matplotlib.pyplot as plt
import numpy as np

def encode_image(support_img, img_to_hide):
    # Display the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Support Image')
    plt.imshow(support_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image to Hide')
    plt.imshow(img_to_hide)
    plt.axis('off')

    plt.show()

    # Make sure the images are the same dimension
    support_img = support_img[:img_to_hide.shape[0], :img_to_hide.shape[1]]
    original_support_img = support_img.copy()
    original_hide_img = img_to_hide.copy()

    # Put zeros to all the 4 less significant bits of the support image and display it (comparing with the original one)
    support_img = support_img & 0b11110000
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Support Image with 4 less significant bits set to 0')
    plt.imshow(support_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Original Support Image')
    plt.imshow(original_support_img)
    plt.axis('off')

    plt.show()

    # Shift to the right the 4 most significant bits of the image to hide and display it (comparing with the original one)
    img_to_hide = img_to_hide >> 4
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Image to Hide with 4 most significant bits shifted to the right')
    plt.imshow(img_to_hide)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Original Image to Hide')
    plt.imshow(original_hide_img)
    plt.axis('off')

    plt.show()

    # Build a new image by adding the two images and display it
    stego_img = support_img + img_to_hide
    plt.figure(figsize=(10, 5))

    plt.title('Stego Image')
    plt.imshow(stego_img)
    plt.axis('off')

    plt.show()

    # Save it as png
    skimage.io.imsave('imgs/stego_img.png', stego_img)

def decode_image(mixed_img):
    # Get the 4 less significant bits of the image
    bits = mixed_img & 0b00001111

    # Shift to the left the 4 bits to get the original image
    original_img = bits << 4

    # Display the original image
    plt.figure(figsize=(10, 5))
    plt.title('Original Image')
    plt.imshow(original_img)
    plt.axis('off')
    plt.show()

def test_decode():
    print(f"Testing decode_image() function on mixed_img1")
    decode_image(skimage.io.imread('imgs/mixed_img1.png'))
    print(f"Testing decode_image() function on mixed_img2")
    decode_image(skimage.io.imread('imgs/mixed_img2.png'))

def main():
    try:
        # Load the images
        support_img = skimage.io.imread('imgs/support_img.png')
        img_to_hide = skimage.io.imread('imgs/img_to_hide.png')

        print(np.min(support_img), np.max(support_img))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    encode_image(support_img, img_to_hide)

    try:
        encoded_img = skimage.io.imread('imgs/stego_img.png')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    decode_image(encoded_img)

    test_decode()

if __name__ == "__main__":
    main()
