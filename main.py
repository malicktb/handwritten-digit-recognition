from data_loader import load_data
from knn_model import build_knn_model
from cnn_model import CNN, load_mnist_data, train_model, predict_custom_image
from image_utils import show_image, load_custom_image
import random
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    # Ask the user to choose between KNN and CNN
    model_choice = input("Choose a model (KNN/CNN): ").strip().upper()

    if model_choice == "KNN":
        # Load dataset for KNN
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
        image_filename = input("Enter the name of your image file (e.g., 'eightimage.png'): ")

        # Load the custom image for testing
        img = load_custom_image(image_filename)
        plt.imshow(img, cmap=plt.cm.gray_r)

        # Predict custom image using KNN
        predicted_custom = knn_model.predict(img.flatten().reshape(1, -1))
        print(f"Predicted for custom image using KNN: {predicted_custom}")

    elif model_choice == "CNN":
        # Load dataset for CNN
        train_loader, val_loader, test_loader = load_mnist_data(batch_size=32)

        # Initialize the CNN model
        cnn_model = CNN()

        # Train the CNN
        print("Training CNN...")
        train_model(cnn_model, train_loader, val_loader, num_epochs=10)

        # Visualize a random sample from the test set
        which = random.randint(0, len(test_loader.dataset) - 1)
        test_images, test_labels = next(iter(test_loader))
        plt.imshow(test_images[which][0], cmap=plt.cm.gray_r)
        plt.show()

        # Predict a random sample using CNN
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(test_images[which].unsqueeze(0))
            _, predicted = output.max(1)
        print(f"Predicted: {predicted.item()}, Actual: {test_labels[which].item()}")

        # Prompt user for a custom image file
        image_filename = input("Enter the name of your image file (e.g., 'eightimage.png'): ")

        # Load the custom image for testing
        img = load_custom_image(image_filename)
        plt.imshow(img, cmap=plt.cm.gray_r)

        # Reshape and normalize the custom image to fit the CNN input
        img_tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0) / 255.0

        # Predict custom image using CNN
        predicted_custom = predict_custom_image(cnn_model, img_tensor)
        print(f"Predicted for custom image using CNN: {predicted_custom}")

    else:
        print("Invalid choice. Please choose either 'KNN' or 'CNN'.")