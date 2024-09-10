from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier  # For CNN later

def build_knn_model(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

def build_cnn_model():
    # Placeholder for CNN implementation (using MLPClassifier or a deep learning library)
    pass