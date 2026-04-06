import numpy as np

class ReLU :

    def forward_pass(self ,z):
        self.z = z
        return np.maximum(0,z)

    def backward_pass(self , dA):
        return dA * (self.z>0).astype(float)

class Sigmoid:

    def forward_pass(self,z):

        self.a = 1 /(1 + np.exp(-z))  
        return self.a

    def backward_pass(self,dA):

        return dA*(self.a*(1-self.a))              

class Softmax :
     
     def forward_pass(self , z):

        expz = np.exp(z - np.max(z, axis = 1, keepdims = True))
        self.a = expz/np.sum(expz,axis=1,keepdims = True)

        return self.a

     def backward_pass(self,dA):
        return dA   
    # Gradient of softmax with cross-entropy simplifies to (y_pred - y_true)
        """
        In pytorch, tensorflow the softmax activation
        is calculate using jacobians. here we use simple not using that
        complex method.
       """

         