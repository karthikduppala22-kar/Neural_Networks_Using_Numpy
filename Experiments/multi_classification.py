import numpy as np
import matplotlib.pyplot as plt


from Neural_Network.layer import Layer
from Neural_Network.activation import ReLU, Softmax
from Neural_Network.loss import CategoricalCrossEntropy
from Neural_Network.model import NeuralNetwork

from Optimizers.adam import Adam
from train import train

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = load_wine()
X = data.data
y = data.target


encoder = OneHotEncoder(sparse_output=False)# its is for convert categorical labels into binary  matrix format
y = encoder.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# building the model
model = NeuralNetwork([
    Layer(13,8),# 13 feature in dataset
    ReLU(),
    Layer(8,3),
    Softmax()
])

# for multi-class classification we use categorical cross-entropy 
loss_fn = CategoricalCrossEntropy()
optimizer = Adam(model.get_params(), lr=0.001)


#traning the model
losses = train(model, optimizer, loss_fn, X, y, epochs=100)


y_pred = model.forward_pass(x_test)

pred_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(pred_classes == true_classes)
print("Accuracy:", accuracy)


import matplotlib.pyplot as plt
# 🔹 Plot
plt.plot(losses)
plt.title("Multi-class Classification (wine dataset)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()