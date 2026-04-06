class NeuralNetwork :

    def __init__(self,x):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward_pass(dA)

    def get_params(self): # return the parameters of the network beacuse we need to update them using optimizers
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "parameters"):
                layer_params = layer.parameters()
                for key, value in layer_params.items():
                    params[f"{key}{i}"] = value
        return params                

    def get_grads(self): # return the gradients of the network beacuse we need to update them using optimizers
        grads = {}
        for i,layer in enumerate(self.layers):
            if hasattr(layer, "gradients"):
                layer_grads = layer.gradients()
                for key, value in layer_grads.items():
                    grads[f"{key}{i}"] = value    
        return grads