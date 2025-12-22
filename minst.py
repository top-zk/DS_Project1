import os
from os.path import split
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


MNIST_PATH = r"D:\mnist\mnist_organized\train"
def load_mnist(mnist_path):
    lable= {}
    Data = []
    Label = []
    for i,j in enumerate(os.listdir(mnist_path)):
        lable[j] = i
        for image_name in os.listdir(mnist_path+'/'+j):
            image_path= os.path.join(mnist_path+'/'+j, image_name)
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Data.append(image)
            Label.append(lable[j])

    Data = np.array(Data)
    Label = np.array(Label)
    np.save("Data.npy", Data)
    np.save("Label.npy", Label)
    return Data, Label


def split_data(Data,Label,p):

    index = np.arange(Data.shape[0])
    random_index= np.random.permutation(index)
    train_index= random_index[0:np.floor(p*len(index)).astype(int)]
    val_index= random_index[np.floor(p*len(index)).astype(int):]
    train_data= Data[train_index]
    train_label= Label[train_index]
    val_data= Data[val_index]
    val_Label= Label[val_index]
    return train_data,train_label,val_data,val_Label

Data, Label = load_mnist(MNIST_PATH)
train_data,train_label,val_data,val_Label= split_data(Data,Label,p=0.8)

try:
    Data = np.load("Data.npy")
    Label = np.load("Label.npy")
except:
    mnist_path = r"D:\mnist\mnist_organized\train"
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

