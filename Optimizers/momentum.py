import numpy as np

class Momentum:
    
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        
        # velocity
        self.v = {}
        for key in params:
            self.v[key] = np.zeros_like(params[key])
    
    def step(self, grads):
        for key in self.params:
            dkey = "d" + key
            
            # update velocity
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * grads[dkey]
            
            # update params
            self.params[key] -= self.lr * self.v[key]