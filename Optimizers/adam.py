import numpy as np

class Adam :

    def __init__(self,params,lr=0.01,beta1=0.9,beta2=0.999,eps=1e-8):

        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = {}
        self.v = {}

        for key in params :

            self.m[key]= np.zeros_like(params[key])
            self.v[key]= np.zeros_like(params[key])

    def step(self,grads):

        self.t += 1

        for key in self.params:

            dkey = "d" + key
            dw = grads[dkey]   

            self.m[key] = self.m[key] * self.beta1 + (1 - self.beta1)*dw
            self.v[key] = self.v[key] * self.beta2 + (1 - self.beta2)*(dw**2)

            m_cap = self.m[key] / (1 - self.beta1**self.t)
            v_cap = self.v[key] / (1 - self.beta2**self.t)

            self.params[key] -= self.lr * (m_cap / (np.sqrt(v_cap) + self.eps))