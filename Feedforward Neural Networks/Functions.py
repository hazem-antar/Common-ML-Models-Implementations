import numpy as np
import matplotlib.pyplot as plt

def visualize_performance(train_losses, validation_losses, loss_type = "BCE"):
    '''
    Function that visualizes the training and validation loss over epochs.
    - train_losses: A list of loss values for each epoch during training.
    - validation_losses: A list of loss values for each epoch during validation.
    '''

    # Determine the number of epochs
    epochs = range(len(train_losses))

    # Plot the training and validation losses
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, validation_losses, label='Validation Loss', color='orange')

    # Calculate minimum loss values and corresponding epochs
    min_train_loss = np.min(train_losses)
    min_val_loss = np.min(validation_losses)
    epoch_min_train_loss = np.argmin(train_losses)
    epoch_min_val_loss = np.argmin(validation_losses)

    # Highlight the epochs with the minimum training and validation losses
    plt.scatter(x = epoch_min_train_loss, y = min_train_loss, color='blue', s = 20, 
                label = f'Min Training Loss: {min_train_loss:.4f}')
    plt.scatter(x = epoch_min_val_loss, y = min_val_loss, color='orange', s = 20, 
                label = f'Min Validation Loss: {min_val_loss:.4f}')
    
    # Set the number of ticks
    plt.xticks(np.arange(0, len(epochs), max(int(len(epochs)/10), 1)))

    # Add a title and axis labels
    plt.title(f"Training and Validation ({loss_type}) Loss Over Epochs")
    plt.xlabel("Training Epoch")
    plt.ylabel(f"Loss ({loss_type})")

    # Add a grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legends
    plt.legend()

    # Display the plot
    plt.show()

def activation(Z, type = "relu"):
    '''
    Collective function that does different non-linear activations
    '''
    if type == "relu":
        h = np.maximum(0, Z)
    elif type == "tanh":
        h = np.tanh(Z)
    elif type == "sigmoid":
        h = 1 / (1 + np.exp(-Z))
    elif type == "softmax":
        expZ = np.exp(Z - np.max(Z))
        h = expZ / expZ.sum(axis=1, keepdims=True)
    return h

def activation_derivative(Z, type = "relu"):
    '''
    Collective function that calculate the derivative of different non-linear activations
    '''
    if type == "relu":
        h = np.where(Z > 0, 1, 0)
    elif type == "tanh":
        h = 1 - activation(Z, "tanh")**2
    elif type == "sigmoid":
        h = activation(Z, "sigmoid") * (1 - activation(Z, "sigmoid"))
    elif type == "softmax":
        S = activation(Z, "softmax")
        S_diag = np.zeros((S.shape[0], S.shape[1], S.shape[1]))
        np.einsum('ij,jk->ijk', S, np.eye(S.shape[1]), out=S_diag)
        h = S_diag - S[:,:,None] * S[:,None,:]
    return h


class Feed_Forwards_Neural_Netowrk():

    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes, hidden_activation = 'relu',
                  output_activation = 'sigmoid', loss = "BCE", w_init_type = 'He') -> None:
        
        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.n_hidden_layers = len(n_hidden_nodes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss

        # Weights initialization
        layer_sizes = [n_input_nodes] + n_hidden_nodes + [n_output_nodes]
        self.Initialize_Weights(layer_sizes, type = w_init_type)

    def Initialize_Weights(self, layer_sizes, type = "He") -> None:
        '''
            Method that initializes the weights of the neural network in one of different ways.
            Layer_sizes: Input list containing the number of nodes in input_layer, hidden layer/s,
            and output layer respectively.
            type: Input string indicating the type of weights initialization required.
        '''
        self.weights = []

        if type == "He":  # If "He" Initialization
            for i in range(1, len(layer_sizes)):

                # Weights initialization
                W = np.random.randn(layer_sizes[i-1] + 1, layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1])
                
                # Bias initializations (first column)
                W[:, 0] = 0

                self.weights.append(W)

    def Forward(self, X) -> np.ndarray:
        '''
        Method that performs a forward pass in the Neural Network using the input features
        X: Input features
        Y: output predictions
        '''

        # Initialize lists to store the z's and h's for backpropagation
        self.z_list, self.h_list = [], []
        
        # Input X
        h = X

        # Iterate over hidden layers
        for i in range(self.n_hidden_layers): 

            # Adding a column of 1's at the beginning of h to account for the biases
            h = np.hstack([np.ones((h.shape[0], 1)), h])

            # Apply linear combination of the inputs and weights + bias
            z = np.dot(h, self.weights[i])

            # Apply non-linear activation
            h = activation(z, self.hidden_activation)  
            
            # Store the z's and h's 
            self.z_list.append(z)
            self.h_list.append(h)

        # Output layer
        h = np.hstack([np.ones((h.shape[0], 1)), h])
        z = np.dot(h, self.weights[-1]) 
        Y = activation(z, self.output_activation)  
        self.z_list.append(z)
        return Y
    
    def binary_cross_entropy_loss(self, Y_true, Y_pred):
        '''
        Method that calculates Binary Cross-Entropy Loss.
        Y_true: Array of true labels (0 or 1).
        Y_pred: Array of predicted probabilities.
        '''
        Y_pred = np.clip(Y_pred, 1e-9, 1 - 1e-9)
        m = Y_true.shape[0]
        
        # Original loss calculation (without regularization)
        original_loss = -1/m * np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))

        # total loss
        loss = original_loss

        return loss

    def Backward(self, t, y, x) -> None:
        '''
        Method that perform backward propagation and update weights and biases. 
        t: True labels
        y: Predictions from the forward pass
        '''
        self.gradients = []

        # 1) For the output layer: ---

        # Gradient of loss w.r.t. y 
        if self.loss == "BCE":  # Binary cross entropy (for binary classification)
            L_wrt_y = (y - t) / ((y * (1 - y)) + 1e-9)

        # Derivative of y w.r.t. last layer output z_d
        y_wrt_zd = activation_derivative(self.z_list[-1], self.output_activation)

        # Gradient of loss w.r.t. last layer output z_d
        L_wrt_zd = L_wrt_y * y_wrt_zd
 
        # 2) For ouput and all hidden layers except first: ---

        # Initialize loss w.r.t. layer j as layer d
        L_wrt_zj = L_wrt_zd

        for j in reversed(range(1, self.n_hidden_layers+1)):

            # Gradient of loss w.r.t. layer j weights w_j
            L_wrt_wj = np.dot(np.hstack([np.ones((self.h_list[j-1].shape[0], 1)), self.h_list[j-1]]).T,
                               L_wrt_zj) / x.shape[0]

            # Store Gradient of loss w.r.t. w_j 
            self.gradients.insert(0, L_wrt_wj)     

            # Gradient of loss w.r.t. previous layer output z_(j-1)
            L_wrt_zj = activation_derivative(self.z_list[j-1], self.hidden_activation) *\
            np.dot(L_wrt_zj, self.weights[j][1:, :].T)

        # 3) For first hidden layer 
        L_wrt_w1 = np.dot(np.hstack([np.ones((x.shape[0], 1)), x]).T, L_wrt_zj) / x.shape[0]
        self.gradients.insert(0, L_wrt_w1)     

    def step(self, lr = 0.001, lamda = 0):
        '''
        Method that updates the neural network parameters based on the last calculated gradients
        '''
        for w, w_grad in zip(self.weights, self.gradients):
            
            if (lamda):
                # Updating the biases (no regularization applied to biases)
                w[:, 0] -= lr * w_grad[:, 0] 

                # Updating the weights with regularization (if lambda is set)
                w[:, 1:] -= lr * (w_grad[:, 1:] + 2 * lamda * w[:, 1:])
            else:
                w -= lr * w_grad 

    def data_loader(self, X, Y, batch_size):
        '''
        Method that generates batches of training data.
        '''
        n_samples = X.shape[0]

        # Shuffle the data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield X[batch_indices], Y[batch_indices]
            
    def train(self, X_train, Y_train, X_validation, Y_validation, n_epochs = 1000,
              optimizer = "MB-SGD", batch_size = 128, lr = 0.01, lamda = 0,
              early_stop = False, patience = 10):
        '''
        Method that trains the neural network by performing forward passes, backpropagation,
          and network updates for a specified number of epochs.
        '''
        # Lists to store the losses across training
        train_losses, validation_losses = [], []

        # Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):

            # Determine the batch size based on the optimizer
            if optimizer == "SGD":
                current_batch_size = 1
            elif optimizer == "MB-GD":
                current_batch_size = batch_size
            elif optimizer == "B-GD":
                current_batch_size = X_train.shape[0]

            # Phase 2: Train ---
            
            # List to store the loss of each training batch
            batch_losses = []

            # Generate batches on the fly, more memory efficent
            for X_batch, Y_batch in self.data_loader(X_train, Y_train, current_batch_size):

                # Forward pass on the training data
                predictions_train = self.Forward(X_batch)

                # Calculate training loss
                if self.loss == "BCE":  # Binary cross entropy (for binary classification)
                    batch_losses.append(self.binary_cross_entropy_loss(Y_batch, predictions_train))
                
                # Backward pass 
                self.Backward(Y_batch, predictions_train, X_batch)

                # Network Update
                self.step(lr = lr, lamda = lamda)
            
            # Phase 3: Validate ---

            # Forward pass on the validation data
            predictions_validation = self.Forward(X_validation)
            
            # Calculate validation loss
            if self.loss == "BCE":  # Binary cross entropy (for binary classification)
                loss_validation = self.binary_cross_entropy_loss(Y_validation, predictions_validation)
            
            # Phase 3: Bookeeping ---
                
            # Log the losses
            train_losses.append(np.mean(batch_losses))
            validation_losses.append(loss_validation)

            # Early stopping check
            if early_stop:
                if loss_validation < best_loss:
                    best_loss = loss_validation
                    epochs_no_improve = 0  # Reset counter
                else:
                    epochs_no_improve += 1  # Increment counter
                if epochs_no_improve >= patience:
                    break  # Stop training

        return train_losses, validation_losses
