import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.stats import mode

def TrainTest_Bagging(X_train, X_test, y_train, y_test, n_predictors = 50):
    '''
    Custom-made Implementation of Bagging Ensamble Method
    Inputs: Training and testing features and targets, along with the hyperparameters
    Outputs: Testing missclassification rate (0/1 loss) of the ensamble after training
    '''
    BasePredictors = []

    # Training Base Predictors
    for p in range(n_predictors):

        # Define a base classifiers (Decision Tree)
        baseTree = DecisionTreeClassifier()

        # Draw N (size of training set) examples from X_train and y_train, with repetition
        X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=X_train.shape[0])
        
        # Fit the base classifier
        baseTree.fit(X_train_sample, y_train_sample)

        # Store base classifier
        BasePredictors.append(baseTree)

    # Testing each predictor (could also be done on the fly inside training for loop without
    # the need for storing each predictor but I wanted to simulate that we are saving the model)
    predictions = np.array([baseTree.predict(X_test) for baseTree in BasePredictors])
    
    # Take majority vote of the ensamble
    y_pred = mode(predictions, axis=0)[0]

    # Calculate the misclassification rate
    loss = np.mean(y_test != y_pred)

    return loss

def TrainTest_RandomForest(X_train, X_test, y_train, y_test, n_predictors = 50):
    '''
    Premade SKlearn Implementation of RandomForest Ensamble Method
    Inputs: Training and testing features and targets, along with the hyperparameters
    Outputs: Testing missclassification rate (0/1 loss) of the ensamble after training
    '''
    # Initialize the model
    RandomForest_Model = RandomForestClassifier(n_estimators=n_predictors)

    # Train the model
    RandomForest_Model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = RandomForest_Model.predict(X_test)

    # Calculate the misclassification rate
    loss = np.mean(y_test != y_pred)

    return loss

def TrainTest_Adaboost(X_train, X_test, y_train, y_test, n_predictors = 50, baseType = "stump"):
    '''
    Custom-made Implementation of Adaboost Ensemble Method using multiple types of base classifiers
    Inputs: Training and testing features and targets, along with the hyperparameters
    Outputs: Testing missclassification rate (0/1 loss) of the ensamble after training
    '''

    BasePredictors = []

    # Initialize list to store the amount of say (α)
    alphas = []

    # Initialize Weights (β) for all training samples (1/total number of samples)
    sample_weights = np.full(y_train.shape, 1/y_train.shape[0])

    if baseType == "stump":
        # Compute the indices that would sort each column once and use them many tims
        sorted_indices = np.argsort(X_train, axis=0)

    # Training Phase: ------------------------
    for p in range(n_predictors):

        # 1) Fit a base predictor: -------------

        if baseType == "stump":
            basePredictor = Fit_Decision_Stump(X_train, y_train, sorted_indices, sample_weights)

        elif baseType == "tree_10":

            # Initialize a decision tree classifier with at most 10 leaves
            basePredictor = DecisionTreeClassifier(max_leaf_nodes=10)

            # Train the classifier using the weighted samples
            basePredictor.fit(X_train, y_train, sample_weight = sample_weights)
        elif baseType == "tree":

            # Initialize a decision tree classifier with no leaves restrictions
            basePredictor = DecisionTreeClassifier()

            # Train the classifier using the weighted samples
            basePredictor.fit(X_train, y_train, sample_weight = sample_weights)

        # Store the base classifier
        BasePredictors.append(basePredictor)

        # 2) Calculate the total (weighted) loss ϵ and amount of say α of the base predictor: ----------
        
        # Calculate the base classifier predictions on training set
        if baseType == "stump":
            X_pred = Predict_Decision_Stump(stump = basePredictor, X = X_train)
        elif baseType == "tree_10" or baseType == "tree":
            X_pred = basePredictor.predict(X_train)

        # Calculate the total loss (ϵ)
        sigma = np.sum(sample_weights[X_pred != y_train])
        
        # Calculate amount of say (α)
        eps = 1e-10  # A small constant to prevent division by zero
        alpha = 0.5 * np.log((1 - sigma) / (sigma + eps))

        # Store the amount of say
        alphas.append(alpha)

        # 3) Update the sample weights (β): ----------
        sample_weights *= np.exp(-alpha * y_train * X_pred)

        # Normalize the weights to sum to 1
        sample_weights /= np.sum(sample_weights)

        ''' Another method as in lecture but the above is more popular
        
        # Increase the weight of misclassified examples
        sample_weights[X_pred != y_train] /= (2 * sigma)

        # Decrease the weight of correctly classified examples
        sample_weights[X_pred == y_train] /= (2 * (1 - sigma))
        '''

    # Testing Phase: ------------------------
        
    # Initialize a numpy array to accumulate weighted votes
    weighted_votes = np.zeros(len(y_test))

    for basePredictor, alpha in zip(BasePredictors, alphas):

        # Get predictions for the current base predictor
        if baseType == "stump":
            X_pred = Predict_Decision_Stump(stump = basePredictor, X = X_test)
        elif baseType == "tree_10" or baseType == "tree":
            X_pred = basePredictor.predict(X_test)

        # Update weighted votes 
        weighted_votes += alpha * X_pred

    # Determine final predictions (1/-1) based on the sign of weighted_votes
    final_predictions = np.where(weighted_votes >= 0, 1, -1)

    # Calculate the misclassification rate
    loss = np.mean(y_test != final_predictions)

    return loss
    
def Fit_Decision_Stump(X_train, y_train, sorted_indices, weights, num_features_to_consider = None):
    '''
    Custom-made Implementation of Decision Stump that finds the best split across all features
    that minimizes weighted Gini Index impurity.
    Inputs: Training features and targets, sorted indices of every feature, and the sample weights
    of every data point.
    Outputs: dictionary of split_col, split_value, split_labels representing the stump model
    '''
    # Initialize variables
    best_split_col, best_split_value, best_split_index, best_GiniIndex = 0, 0, 0, np.inf

    # Total number of features
    total_features = X_train.shape[1]
    
    # Select the features to search splitting
    if num_features_to_consider == None:
        features_to_consider = range(total_features)
    else:
        features_to_consider = np.random.choice(range(total_features), num_features_to_consider, replace=False)

    # Test splitting all features
    for col_index in features_to_consider: 

        # Get the sorted values and corresponding sorted labels and weights for the current feature
        sorted_values = X_train[sorted_indices[:, col_index], col_index]
        sorted_labels = y_train[sorted_indices[:, col_index]]
        sorted_weights = weights[sorted_indices[:, col_index]]

        # Evaluate splits only at midpoints between consecutive sorted values
        for i in range(1, len(sorted_values)):
            
            # Skip if consecutive values are identical
            if sorted_values[i] == sorted_values[i-1]:
                continue

            # Midpoint between consecutive values
            split_value = (sorted_values[i] + sorted_values[i-1]) / 2

            # Calculate the weighted Gini index of this split (right + left impurities)
            GiniIndex = weighted_gini_index(i, sorted_labels, sorted_weights)
            
            # Update best split information
            if GiniIndex < best_GiniIndex:
                best_split_col= col_index
                best_split_value = split_value
                best_split_index = i
                best_GiniIndex = GiniIndex

    # Extract indices of right and left groups of the best split
    left_indices = sorted_indices[:, best_split_col][:best_split_index]
    right_indices = sorted_indices[:, best_split_col][best_split_index:]

    # Use the indices to extract the weights and labels of the groups
    left_weights, left_labels = weights[left_indices], y_train[left_indices]
    right_weights, right_labels = weights[right_indices], y_train[right_indices]

    # Assign a label to each group based on which internal class has the maximum sum of weights 
    left_label = max(np.unique(left_labels), key=lambda label: np.sum(left_weights[left_labels == label]))
    right_label = max(np.unique(right_labels), key=lambda label: np.sum(right_weights[right_labels == label]))

    # Save the labeling of each group
    best_split_labels = [left_label, right_label]
       
    return {'split_col':best_split_col, 'split_value':best_split_value, 'split_labels':best_split_labels}

def Predict_Decision_Stump(stump, X):
    '''
    Use a decision stump to make a prediction
    Inputs: Features X
    Output: The predicted classes Y
    '''
    # Unpack stump
    split_col, split_value, split_labels = stump.values()

    # Initialize predictios
    predictions = np.zeros(X.shape[0], dtype=int)
    
    # Set predictions according to the split criteria
    predictions[X[:, split_col] < split_value] = split_labels[0]
    predictions[X[:, split_col] >= split_value] = split_labels[1]

    return predictions

def weighted_gini_index(split_index, labels, weights):
    '''
    Calculate the weighted Gini index for a given split.
    Inputs: The split index (in the sorted feature), the sorted labels and weights.
    Outputs: weighted Gini index.
    '''

    # Initialize the weighted Gini index
    total_gini = 0

    # Define the two group indices after splitting
    groups = [np.arange(split_index), np.arange(split_index, len(labels))]
            
    for group in groups:

        # Initialize the group Gini index
        group_gini = 0

        group_weight = weights[group].sum()

        # Ignore empty groups
        if group_weight == 0: continue

        # Calculate proportion of weight for the first class (1)
        p_1 = weights[group][labels[group] == 1].sum() / group_weight

        # Proportion of weight of the second class (-1) is 1 - p_1
        p_2 = 1.0 - p_1

        # Calculate the group's Gini index
        group_gini = 1 - (p_1 ** 2 + p_2 ** 2)

        # Weight the group's Gini index by its proportion of the total weight
        total_gini += group_gini * group_weight

    return total_gini
