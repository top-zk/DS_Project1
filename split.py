from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from Load_mnist import *

try:
    Data = np.load("Data.npy")
    Label = np.load("Label.npy")
except:
    mnist_path = r"C:\Users\student\Downloads\MNIST\MNIST"
    Data, Label, lable = load_mnist(mnist_path)
X_train, X_test, y_train, y_test = split_data(Data, Label, p=0.8)
# Creating the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
# Training the classifier
knn.fit(X_train, y_train)
# Making predictions
y_pred = knn.predict(X_test)
# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
