# Neural Networks from Scratch (NumPy)

This repository contains implementations of neural networks built completely from scratch using NumPy. Not used any of Deep learning framework. Completely training of a neural network from scratch using raw python and numpy. This makes the foundations stronger before learning basic neural networks with out using framework makes us to know about the how internally a DL framework compute the large matrics and gradients.

Initially i have built the neural network using functions in notebooks , But after i have built it like a small framework for any one can use my mini frame work. Using OOPs and modules and made the code more efficient then before . I separated layers, activations, loss functions, optimizers, and training logic into different files to make the code cleaner and reusable.


## Experiments

I tested the model on different types of datasets:

- Binary classification (breast cancer dataset)  
- Multi-class classification ( Wine dataset)  
- MNIST dataset (image classification)

## Results
 - Here I have saved the ploted model learning curve of the model ( losses vs epochs ) 
## Features

- Dense (fully connected) layers  
- Activation functions: ReLU, Sigmoid, Softmax  
- Loss functions: Binary Cross Entropy, Categorical Cross Entropy  
- Optimizers: SGD, Momentum, RMSProp, Adam  
- Mini-batch training  
- Modular neural network structure  

## Tech Stack

* Python
* NumPy
* scikit-learn(for only importing and scaling data)

## Goal

To deeply understand neural networks by implementing everything from scratch without using deep learning frameworks.
