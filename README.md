# Handwritten Digit Recognition

This project implements key ideas from Yann LeCun's paper *Handwritten Digit Recognition with a Back-Propagation Network* using both **K-Nearest Neighbors (KNN)** and **Convolutional Neural Networks (CNN)**.

## Project Structure
- `main.py`: Main entry point to load data, train models (KNN/CNN), and make predictions (including custom images).
- `data_loader.py`: Handles loading and splitting of the dataset.
- `model.py`: Contains different models (KNN and CNN with backpropagation).
- `image_utils.py`: Utility functions for image display, loading, and preprocessing.
- `cnn_model.py`: Defines the CNN architecture, training, and prediction functions.

## Features
- **K-Nearest Neighbors (KNN)**: For basic handwritten digit recognition.
- **Convolutional Neural Networks (CNN)**: Implements CNN with **backpropagation** using **PyTorch** for improved accuracy and performance.
- **Custom Image Prediction**: Load and predict handwritten digits from user-supplied images.

## Future Improvements
- Experimenting with additional data augmentation techniques to improve robustness.
