import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x_train, y_train, x_valid, y_valid, max_iter=10000, learning_rate=0.001, lamda = 0):
    '''
    Custom-made Implementation for Training a Logistic Regression Model
    Inputs: Training and validation features and targets, along with the hyperparameters
    Plots: A curve showing the training and validation loss over training iterations
    Outputs: Weights and bias vectors of the trained model
    '''
    # Weights and bias vectors are initialized to be zero
    weights = np.zeros((x_train.shape[1], 1))
    bias = np.zeros(1)

    # Total number of training samples
    N = x_train.shape[0]

    # Lists to store the training and validaiton losses
    train_losses = []
    valid_losses = []

    for i in range(max_iter):

        # Calculate the training and validation predictions
        train_predictions = sigmoid(np.dot(x_train, weights) + bias)
        valid_predictions = sigmoid(np.dot(x_valid, weights) + bias)

        # Claculate trainning Cross-entropy loss
        train_loss = -np.mean(y_train * np.log(train_predictions) + (1 - y_train) * np.log(1 - train_predictions))
        train_losses.append(train_loss)

        # Claculate validation Cross-entropy loss
        valid_loss = -np.mean(y_valid * np.log(valid_predictions) + (1 - y_valid) * np.log(1 - valid_predictions))
        valid_losses.append(valid_loss)

        # Compute the gradient of the cost function
        gradient_weights = (np.dot(x_train.T, (train_predictions - y_train))/N) + 2 * lamda * weights
        gradient_bias = np.mean(train_predictions - y_train, axis=0)

        # Make a gradient decent step
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    # Plot the losses
    plt.plot(np.arange(max_iter), train_losses, label = "Training Loss", linewidth=1)
    plt.plot(np.arange(max_iter), valid_losses, label = "Validation Loss", linewidth=1)

    # Add legend
    plt.legend()

    # Add title and axis labels
    plt.title("Trainning a Custom Logistic Regression Model")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cross-entropy Loss")

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Increase axis precision
    plt.xticks(np.arange(0, max_iter+1000, 1000))
    plt.yticks(np.arange(0, 0.8, 0.1))

    # Render the plot
    plt.show()

    return weights, bias

def Classification_Metrics(true_labels, predicted_labels):
    '''
    Function that calculates different classification metrics (scores, rates, losses).
    Input: the actual and the predicted labels. 
    Output: A dictionary containing the different calculated metrics.
    '''
    # Calculate TP, FP, TN, FN
    TP = np.sum((predicted_labels == 1) & (true_labels == 1))
    FP = np.sum((predicted_labels == 1) & (true_labels == 0))
    TN = np.sum((predicted_labels == 0) & (true_labels == 0))
    FN = np.sum((predicted_labels == 0) & (true_labels == 1))
    
    # Calculate misclassification rate
    misclassification_rate = (FP + FN) / len(true_labels)

    # Calculate the (recall / sensitivity / TP_rate)
    recall = TP / (TP + FN) 
    
    # Calculate the FP_rate
    FP_rate = FP / (TN + FP)

    # Calculate the specificity
    specificity = 1 - FP_rate

    # Calculate precision
    precision = TP / (TP + FP) 

    # Calculate F1 score
    F1 = 2 * (precision * recall) / (precision + recall)
    
    return {"TP_rate":recall, "FP_rate":FP_rate, "precision":precision,
             "recall":recall, "sensitivity":recall, "specificity":specificity,
             "misclassification_rate":misclassification_rate, "F1_score":F1}


def Draw_ROC(true_labels, predictions, title):
    '''
    Function that draws the ROC curve of a given model using the sorted probability predictions as the thresholds.
    Input: The true labels and the predictions (raw probabilities)
    Plots: The ROC curve over the different thresholds
    Output: The misclassification rate and F1 score for the optimal threshold.
    '''

    # Lists to store the TP nad FP rates for the ROC
    TP_rates = []
    FP_rates = []

    # Lists to store the misclassification_rates and F1_scores
    misclassification_rates = []
    F1_scores = []

    # Evaluate all possible thresholds based on the test dataset predictions
    for threshold in np.sort(predictions):

        # Apply thresholding
        predicted_labels = predictions >= threshold

        # Calculate Classification Metrics
        metrics =  Classification_Metrics(true_labels, predicted_labels)
        
        # Store the needed metrics
        TP_rates.append(metrics["TP_rate"])
        FP_rates.append(metrics["FP_rate"])
        misclassification_rates.append(metrics["misclassification_rate"])
        F1_scores.append(metrics["F1_score"])

    # Plot a baseline diagonal from (0,0) to (1,1)
    plt.plot([0, 1], [0, 1], color='orange', linewidth=1, label = 'baseline', linestyle='--') 

    # Scatter the collected data
    plt.scatter(FP_rates, TP_rates, s=5, color='blue', label = 'Classifiers')
    
    # Find the classifier with minimum misclassification rate
    best_classifier_index = np.argmin(misclassification_rates)
    highlight_x = FP_rates[best_classifier_index]
    highlight_y = TP_rates[best_classifier_index]

    # Highlight it in the graph
    plt.plot([highlight_x, highlight_x], [highlight_y, 0], color='black', linestyle='--', linewidth=1)
    plt.plot([highlight_x, 0], [highlight_y, highlight_y], color='black', linestyle='--', linewidth=1)
    plt.scatter(FP_rates[best_classifier_index], TP_rates[best_classifier_index], color='red', s=10, label = 'Classifier with minimum misclassification rate')

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Increase axis precision
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Add axis labels for clarity
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Add title
    plt.title(title)

    # Add legend
    plt.legend()

    # Render plot
    plt.show()

    # Return the misclassification rate and F1 score of the optimal model
    return misclassification_rates[best_classifier_index], F1_scores[best_classifier_index]


def euclidean_distance(x1, x2):
    '''
    Function to calculate the euclidean distance between two D-dimentional points
    '''
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, X_test, k):
    '''
    Custom-made Implementation for training a KNN model and making a batch of predictions
    Inputs: Training features and targets and testing features, along with the hyperparameter (k)
    Outputs: Batch of predictions for the testing features.
    '''

    # Shuffle the train data to solve the tie problem
    shuffled_indices = np.arange(X_train.shape[0])
    np.random.shuffle(shuffled_indices)
    X_train_shuffled = X_train[shuffled_indices]
    y_train_shuffled = y_train[shuffled_indices]

    # Empty array to store the pridictions
    y_pred = np.empty(X_test.shape[0])

    for i, test_sample in enumerate(X_test):

        # Calculate distances from the test sample to all training samples
        distances = np.array([euclidean_distance(test_sample, x) for x in X_train_shuffled])

        # Get the indices of the k smallest distances
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k closest samples
        k_nearest_labels = y_train_shuffled[k_indices]

        # Determine the most common class label among the k closest samples
        y_pred[i] = mode(k_nearest_labels)[0][0]

    return y_pred

def KFold_Custom(dataset_size, num_folds):
    '''
    A generator function that iteratively gives the train and validation
    indices of a given number of k-folds cross-validation.
    '''
    # The number of data points in each fold
    fold_size = dataset_size // num_folds

    for fold in range(num_folds):

        # Start and enf of validation fold
        valid_start = fold * fold_size
        valid_end = (fold + 1) * fold_size

        # Validation fold indicies
        valid_indices = np.arange(valid_start, valid_end)

        # train folds indicies (all other points except validation)
        train_indices = np.concatenate([np.arange(0, valid_start), np.arange(valid_end, dataset_size)])

        yield train_indices, valid_indices