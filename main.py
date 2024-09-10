from data_loader import load_data
from model import build_knn_model
from image_utils import show_image, load_custom_image
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load dataset
    X_train, X_test, y_train, y_test, digits = load_data()

    # Visualize one of the digit samples
    which = random.randint(0, len(digits.images) - 1)
    show_image(digits.images[which])

    # Train the KNN model
    knn_model = build_knn_model(X_train, y_train)

    # Predict a random sample
    predicted = knn_model.predict(X_test[which].reshape(1, -1))
    print(f"Predicted: {predicted}, Actual: {y_test[which]}")

    # Prompt user for a custom image file
    image_filename = input("enter the name of your image file (e.g., 'eightimage.png'): ")

    # Load the custom image for testing
    img = load_custom_image(image_filename)
    plt.imshow(img, cmap=plt.cm.gray_r)

    # Predict custom image
    predicted_custom = knn_model.predict(img.flatten().reshape(1, -1))
    print(f"Predicted for custom image: {predicted_custom}")