from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def load_data():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)
    return X_train, X_test, y_train, y_test, digits