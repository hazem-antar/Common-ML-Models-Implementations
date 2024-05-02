from  Functions import *
import numpy as np
import matplotlib.pyplot as plt
import csv

# Seed the pseudo number generator (My ID ends with 0393)
np.random.seed(3093)

# Read the dataset to numpy array
dataset = np.genfromtxt("spambase.data", delimiter=",")

_, unique_indices = np.unique(dataset, axis=0, return_index=True)
dataset_unique = dataset[np.sort(unique_indices)]

print(f"Dropped {dataset.shape[0] - dataset_unique.shape[0]} duplicate rows")

# Separate features and targets
X, Y = dataset_unique[ : , : -1], dataset_unique[ : , -1].astype(int)
Y = Y[:, np.newaxis]  # Add extra dimension

# 1) Splitting the dataset into train, validation, and test sets (60% / 20% / 20%): ----

# Shuffel the dataset indicies
num_rows = X.shape[0]
shuffle_indices = np.random.permutation(num_rows)

# Get indices for splitting the dataset
train_end = int(num_rows * 0.6)
validation_end = int(num_rows * 0.8)

# Splitting the data set
X_train, Y_train = X[shuffle_indices[:train_end]], Y[shuffle_indices[:train_end]]
X_validation, Y_validation = X[shuffle_indices[train_end:validation_end]], Y[shuffle_indices[train_end:validation_end]]
X_test, Y_test = X[shuffle_indices[validation_end:]], Y[shuffle_indices[validation_end:]]

# 2) Standardizing the data: ----

# Calculate the mean and standard deviation of the training features
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_std[X_train_std == 0] = 1 # To not division by zero

# Standardize the all three subsets of data
X_train = (X_train - X_train_mean) / X_train_std
X_validation = (X_validation - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# 3) Implementing neural networks without regularization: ---

# List to store the best number of hidden nodes for a given number of layers 
best_hidden_nodes_list = []

for n_hidden_layers in range(1, 4):
    # Initial search space for the number of hidden nodes
    min_hidden_nodes = 8
    max_hidden_nodes = 1023
    
    # Variables to track the best configuration
    best_hidden_nodes = int(max_hidden_nodes/n_hidden_layers)
    best_loss = float('inf')

    # Perform binary search
    while min_hidden_nodes <= max_hidden_nodes:
        
        # Compute the new middle value in the search space
        n_hidden_nodes = (min_hidden_nodes + max_hidden_nodes) // 2

        # Define the size of layers
        n_input_nodes = X.shape[1]
        layers_n_hidden_nodes = n_hidden_layers * [n_hidden_nodes]   # Hidden layers sizes (symmetric) 
        n_output_nodes = 1

        # Instantiate a Neural Network
        FNN = Feed_Forwards_Neural_Netowrk(n_input_nodes, layers_n_hidden_nodes, n_output_nodes,
                                            hidden_activation = "relu", output_activation = "sigmoid",
                                            loss = "BCE", w_init_type = "He")
        
        # Train the Neural Network
        train_losses, validation_losses = FNN.train(X_train, Y_train, X_validation, Y_validation,
                                                    n_epochs = 1000, optimizer = "SGD", lr = 0.001)
        
        # Consider the last training loss
        current_loss = train_losses[-1]  
    
        # Update best configuration if current loss meets the target
        if current_loss < 0.01:
            best_loss = current_loss
            best_hidden_nodes = n_hidden_nodes
        
        # Adjust search space based on the target
        if current_loss < 0.01:
            max_hidden_nodes = n_hidden_nodes - 1  # Try fewer nodes
        else:
            min_hidden_nodes = n_hidden_nodes + 1  # Need more nodes

        # For visualizing a plotting only
        #visualize_performance(train_losses, validation_losses, loss_type = "BCE")

    # Store the best found number of hidden nodes for this number of layers 
    best_hidden_nodes_list.append(best_hidden_nodes)

    # Final evaluation
    print(f"Best #hidden nodes for {n_hidden_layers} hidden layer/s is {best_hidden_nodes} with a training loss of {best_loss}")

# 4) Implementing neural networks with regularization and early stopping: ---

# Retrieving the best found number of hidden nodes
best_hidden_nodes_list = [770, 72, 55]

best_lamdas_list = []

for n_hidden_layers, n_hidden_nodes in zip(range(1, 4), best_hidden_nodes_list):

    # Initialize the lambda search space
    best_lambda = None
    best_validation_loss = float('inf')
    lambda_values = np.logspace(-10, -1, num = 100)

    # Do a grid search for the optimal λ 
    for lamda in lambda_values:

        # Define the size of layers
        n_input_nodes = X.shape[1]
        layers_n_hidden_nodes = n_hidden_layers * [n_hidden_nodes]   # Hidden layers sizes (symmetric)
        n_output_nodes = 1

        # Instantiate a Neural Network
        FNN = Feed_Forwards_Neural_Netowrk(n_input_nodes, layers_n_hidden_nodes, n_output_nodes,
                                            hidden_activation = "relu", output_activation = "sigmoid",
                                            loss = "BCE", w_init_type = "He")
        
        # Train the Neural Network
        train_losses, validation_losses = FNN.train(X_train, Y_train, X_validation, Y_validation,
                                                    n_epochs = 1000, optimizer = "SGD",
                                                    lr = 0.001, lamda = lamda, early_stop = True)
        
        #visualize_performance(train_losses, validation_losses, loss_type = "BCE")
        
        # Update the best lambda if the current model is better
        current_validation_loss = validation_losses[-1]
        if current_validation_loss < best_validation_loss:
            best_lambda = lamda
            best_validation_loss = current_validation_loss  

    # Store the best found lambda
    best_lamdas_list.append(best_lambda)

    # Final evaluation
    print(f"Best lambda for {n_hidden_layers} hidden layer(s) and {n_hidden_nodes} nodes: {best_lambda} with validation loss: {best_validation_loss}")


# 5) Final testing: ---

# Retrieving the best found number of hidden nodes
best_hidden_nodes_list = [770, 72, 55]

# Retrieving the best found lambdas
best_lamdas_list = [1.207e-08, 3.556e-07, 4.498e-10]


for n_hidden_layers, (n_hidden_nodes, lamda) in zip(range(1, 4), zip(best_hidden_nodes_list, best_lamdas_list)):

    # Define the size of layers
    n_input_nodes = X.shape[1]
    layers_n_hidden_nodes = n_hidden_layers * [n_hidden_nodes]   # Hidden layers sizes (symmetric)
    n_output_nodes = 1

    # Instantiate a Neural Network
    FNN = Feed_Forwards_Neural_Netowrk(n_input_nodes, layers_n_hidden_nodes, n_output_nodes,
                                        hidden_activation = "relu", output_activation = "sigmoid",
                                        loss = "BCE", w_init_type = "He")
    
    # Train the Neural Network
    train_losses, validation_losses = FNN.train(X_train, Y_train, X_validation, Y_validation,
                                                n_epochs = 1000, optimizer = "SGD",
                                                lr = 0.001, lamda = lamda, early_stop = True)
        
    # Forward pass on the testing data
    predictions_train = FNN.Forward(X_test)

    # Convert predicted probabilities to class labels
    predicted_labels = (predictions_train >= 0.5).astype(int)
    
    # Print the misclassification rate
    print(f"For a model with {n_hidden_layers} hidden layers, the  misclassification rate is", np.mean(predicted_labels != Y_test))