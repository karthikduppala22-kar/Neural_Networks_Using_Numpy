import numpy as np

class SGD:
    
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self, grads):
        for key in self.params:
            dkey = "d" + key
            self.params[key] -= self.lr * grads[dkey]