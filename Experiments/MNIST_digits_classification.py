import numpy as np
import matplotlib.pyplot as plt


from Neural_Network.layer import Layer
from Neural_Network.activation import ReLU, Softmax
from Neural_Network.loss import CategoricalCrossEntropy
from Neural_Network.model import NeuralNetwork

from Optimizers.adam import Adam
from train import train

from keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784)
X_test  = X_test.reshape(-1, 784)

X_train = X_train / 255.0
X_test  = X_test / 255.0



def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

num_classes = 10

y_train = one_hot_encode(y_train, num_classes)
y_test  = one_hot_encode(y_test, num_classes)


X_train = X_train[:10000]
y_train = y_train[:10000]


# building the model
model = NeuralNetwork([
    Layer(784, 128),
    ReLU(),
    Layer(128, 64),
    ReLU(),
    Layer(64, 10),
    Softmax()
])


# loss function and optimizer
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(model.get_params(), lr=0.001)

 # training model
losses = train(
    model,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    epochs=50,
    batch_size=64
)


# ploting
plt.plot(losses)
plt.title("MNIST Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# accuracy
y_pred = model.forward_pass(X_test)

pred_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(pred_classes == true_classes)

print("Accuracy:", accuracy)



