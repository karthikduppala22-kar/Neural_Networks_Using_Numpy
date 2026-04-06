import numpy as np

def predict(model, X):
    
    y_pred = model.forward_pass(X)
    
    # here ,it convert probabilities → class index
    predictions = np.argmax(y_pred, axis=1)
    
    return predictions