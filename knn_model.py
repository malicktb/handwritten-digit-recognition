from sklearn.neighbors import KNeighborsClassifier

def build_knn_model(X_train, y_train):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    return knn

