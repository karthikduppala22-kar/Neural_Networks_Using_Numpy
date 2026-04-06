import numpy as np

class RMSProp:
    
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        
        # squared gradients
        self.v = {}
        for key in params:
            self.v[key] = np.zeros_like(params[key])
    
    def step(self, grads):
        for key in self.params:
            dkey = "d" + key
            
            # update squared gradients
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grads[dkey] ** 2)
            
            # update params
            self.params[key] -= self.lr * (
                grads[dkey] / (np.sqrt(self.v[key]) + self.eps)
            )