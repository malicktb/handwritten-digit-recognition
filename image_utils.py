import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.show()

def load_custom_image(image_path):
    img = plt.imread(image_path)
    img = np.dot(img[..., :3], [1, 1, 1])  # Convert to grayscale
    img = (8 - img * 8).astype(int)
    return img