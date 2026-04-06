import numpy as np
import matplotlib.pyplot as plt

# importing our modules
from Neural_Network.layer import Layer
from Neural_Network.activation import ReLU, Sigmoid
from Neural_Network.loss import BinaryCrossEntropy
from Neural_Network.model import NeuralNetwork

from Optimizers.adam import Adam
from train import train

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data loading
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1,1) # reshape to make it a column vector

# scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = NeuralNetwork([
    Layer(30,8), # input layer with 30 features and first hidden layer with 8 neurons
    ReLU(),
    Layer(8,4),
    ReLU(),
    Layer(4,1), # output layer with 1 neuron for binary classification
    Sigmoid()
])


# using our defined loss function and optimizer
loss_fn = BinaryCrossEntropy()
optimizer = Adam(model.get_params(), lr=0.001)


# training
losses = train(model, optimizer, loss_fn, X, y, epochs=100)

# testing and finding accuracy
y_pred = model.forward_pass(x_test)

pred_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(pred_classes == true_classes)
print("Accuracy:", accuracy)


# ploting the loss curve
plt.plot(losses)
plt.title("Binary Classification (Adam)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()