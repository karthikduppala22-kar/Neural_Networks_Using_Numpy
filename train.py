import numpy as np

def create_batches(X, y, batch_size): # generator function to create batches of data

    indices = np.arrange(len(X)) 
    np.random.shuffle(indices)
    x = X[indices]
    y = y[indices]
    for i in range(0, len(X), batch_size): 
        yield x[i:i+batch_size], y[i:i+batch_size] # yield is used to create a generator that produces batches of data on-the-fly, which is memory efficient for large datasets.
def train(model , optimizer, loss_fn , x ,y ,epochs = 100 , batch_size = 32, verbose = True):
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for x_batch , y_batch in create_batches(x,y,batch_size):

            y_pred = model.forward_pass(x_batch)
            loss = loss_fn.forward_pass(y_pred,y_batch)

            epoch_loss += loss
            num_batches += 1

            dA = loss_fn.backward_pass()
            model.backward_pass(dA)

            grads =model.get_grads()
            optimizer.step(grads)

        epoch_loss /= max(1, num_batches)
        losses.append(epoch_loss)

        if verbose and  epoch % 10 == 0:
            print(f"Epoch {epoch} , Loss : {epoch_loss:.4f}")    

    return losses      #for ploting the loss curve after training