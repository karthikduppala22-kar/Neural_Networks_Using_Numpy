import numpy as np
class Layer :
    def __init__(self, in_n,out_n):  # here we inintialize parameters required for a neuron

        self.w = np.random.randn(in_n,out_n) * np.sqrt(2/in_n) # He initialization
        self.b = np.zeros((1, out_n))

    def forward_pass(self, A): #input samples primarily, while passing to the hidden layers output of previous neuron
        self.A = A
        self.z = np.dot(A,self.w) + self.b   
        return self.z

    def backward_pass(self,dz):

        m = self.A.shape[0]

        self.dw = (1 / m) * np.dot(self.A.T, dz)
        self.db = (1 / m) * np.sum(dz, axis=0, keepdims=True)

        dA = np.dot(dz, self.w.T)
        return dA
        
    def parameters(self): # return the parameters of the layer
        return {"w":self.w,"b":self.b}

    def gradients(self): # return the gradients of the layer
        return {"dw":self.dw,"db":self.db}    