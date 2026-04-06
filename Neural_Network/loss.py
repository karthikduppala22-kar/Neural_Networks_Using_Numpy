import numpy as np

class BinaryCrossEntropy:
    def forward_pass(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        m = y_true.shape[0]

        loss = -np.mean(
            y_true * np.log(y_pred + 1e-8) +
            (1 - y_true) * np.log(1 - y_pred + 1e-8)
        ) / m

        return loss

    def backward_pass(self):
        # simplified gradient (sigmoid + BCE)
        return self.y_pred - self.y_true

class CategoricalCrossEntropy :

    def forward_pass(self, y_pred,y_true):
        
        self.y_pred = y_pred
        self.y_true = y_true
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward_pass(self):
        return self.y_pred - self.y_true