from  Functions import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
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

# Change class 0 to -1 for easier calculations
Y[Y == 0] = -1

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=42)

# (Q1) Training decision trees -------------------------------

print("\nTraining decision trees --------")

# Set up K-fold cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to track the best score and corresponding number of leaves
best_loss = np.inf
best_n_leaves = 0
losses = []

# Define the range of number of leaves
max_leaves_range = range(2, 401)

# Iterate over all possible number of tree leaves
for n_leaves in max_leaves_range:

    # List to hold the cross-validation losses for a decision tree classifier
    misclassification_rates = []

    for train_indices, valid_indices in kf.split(X_train):
        
        # Split the training dataset (originally 80% of all dataset) to the training and validation folds
        X_train_fold, X_val_fold = X_train[train_indices], X_train[valid_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[valid_indices]
        
        # Initialize a decision tree classifier with the current number of leaves
        possibleTree = DecisionTreeClassifier(max_leaf_nodes=n_leaves)
    
        # Train the classifier
        possibleTree.fit(X_train_fold, y_train_fold)

        # Make predictions
        y_pred = possibleTree.predict(X_val_fold)

        # Append the 0/1 loss (misclassification rate)
        misclassification_rates.append(np.mean(y_val_fold != y_pred))

    # Perform K-fold cross-validation and calculate the average misclassification rate
    average_loss = np.mean(misclassification_rates)

    # Append the new average score
    losses.append(average_loss)

    # Update the least loss so far and the corresponding number of leaves
    if average_loss < best_loss:
        best_loss = average_loss
        best_n_leaves = n_leaves

# Plot the average cross-validation losses against (k)
plt.plot(max_leaves_range, losses)

# Mark the (k) with the minimum average loss
plt.axvline(x = best_n_leaves, color='red', linestyle='--', linewidth=1)

# Increase axis precision
plt.xticks(np.arange(2, 401, 20))

# Add title and axis labels
plt.title("Average Misclassification Rate for Decision Trees using 5-folds Cross-validation")
plt.xlabel("Maximum Number of Tree Leaves")
plt.ylabel("Misclassification Rate (Zero/One Loss)")

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Render Image
plt.show()

print(f"Best maximum number of leaves: {best_n_leaves} with average misclassification rate of: {round(best_loss, 3)}")

# Train a new model on the full training set using the best maximum number of leaves
best_DTree = DecisionTreeClassifier(max_leaf_nodes=best_n_leaves)
best_DTree.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_DTree.predict(X_test)
best_DTree_loss = np.mean(y_test != y_pred)

print(f"The misclassification rate of best decision tree: {round(best_DTree_loss, 3)}")

# -------------------------------------------

print("\nTraining Ensembles --------")

# Define the range of number of predictors in the different ensembles
max_predictors = range(50, 2501, 50)

BaggingClassifierlosses = []
RandomForestlosses = []
AdaboostStumpsClassifier_losses = []
AdaboostTree10Classifier_losses = []
AdaboostTreeClassifier_losses = []

# Create a new CSV to store the losses
with open('ensemble_losses.csv', 'w', newline='') as file:

    # Instantiate a writer
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Number of Predictors', 'Bagging Loss', 'Random Forest Loss', 'AdaBoost Stumps Loss', 'AdaBoost Tree 10 Leaves Loss', 'AdaBoost Unrestricted Tree Loss', 'Decision Tree Loss'])

for n_predictors in tqdm(max_predictors):

    # (Q2) Training a bagging classifier -------------------------------
    BaggingClassifierloss = TrainTest_Bagging(X_train, X_test, y_train, y_test, n_predictors)
    BaggingClassifierlosses.append(BaggingClassifierloss)

    # (Q3) Training a random forest classifier -------------------------------
    RandomForestClassifierloss = TrainTest_RandomForest(X_train, X_test, y_train, y_test, n_predictors)
    RandomForestlosses.append(RandomForestClassifierloss)

    # (Q4) Training Adaboost classifier using decision stumps -------------------------------
    AdaboostStumpsClassifier_loss = TrainTest_Adaboost(X_train, X_test, y_train, y_test, n_predictors, baseType = "stump")
    AdaboostStumpsClassifier_losses.append(AdaboostStumpsClassifier_loss)

    # (Q5) Training Adaboost classifier using decision trees with at most 10 leaves -------------------------------
    AdaboostTree10Classifier_loss = TrainTest_Adaboost(X_train, X_test, y_train, y_test, n_predictors, baseType = "tree_10")
    AdaboostTree10Classifier_losses.append(AdaboostTree10Classifier_loss)

    # (Q6) Training Adaboost classifier using decision trees with no leaves restrictions -------------------------------
    AdaboostTreeClassifier_loss = TrainTest_Adaboost(X_train, X_test, y_train, y_test, n_predictors, baseType = "tree")
    AdaboostTreeClassifier_losses.append(AdaboostTreeClassifier_loss)

    # Open the file in append mode and write the current losses
    with open('ensemble_losses.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([n_predictors, BaggingClassifierloss, RandomForestClassifierloss, AdaboostStumpsClassifier_loss, AdaboostTree10Classifier_loss, AdaboostTreeClassifier_loss, best_DTree_loss])


# Plot the losses of each ensemble type
plt.plot(max_predictors, BaggingClassifierlosses, label="Bagging with Decision Trees")
plt.plot(max_predictors, RandomForestlosses, label="Random Forest")
plt.plot(max_predictors, AdaboostStumpsClassifier_losses, label="AdaBoost with Decision Stumps")
plt.plot(max_predictors, AdaboostTree10Classifier_losses, label="AdaBoost with Trees (max 10 leaves)")
plt.plot(max_predictors, AdaboostTreeClassifier_losses, label="AdaBoost with Unrestricted Trees")

# Plotting a horizontal line at the value best_DTree_loss
plt.axhline(y=best_DTree_loss, color='r', linestyle='--', label="Best Decision Tree Loss")

# Increase axis precision
plt.xticks(np.arange(50, 2501, 200))

# Add title and axis labels
plt.xlabel("Number of Predictors")
plt.ylabel("Misclassification Rate (0/1 Loss)")
plt.title("Classifier Performance Comparison")

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
plt.legend()

# Render Image
plt.show()
