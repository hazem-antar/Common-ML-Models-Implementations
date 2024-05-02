import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from Functions import *

# Seed the pseudo number generator (My ID ends with 0393)
np.random.seed(3093)

# Load the "Wisconsin breast cancer" dataset
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target.astype(int)

# Check class labels
print("Class labels:", list(dataset.target_names))

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# Add extra dimension to the targets
y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Custom Logistic Regression: ------------------------------------

# Train a manual logistic regression model (with regularization)
weights, bias = logistic_regression(X_train, y_train, X_test, y_test, max_iter=10000, lamda = 0.01)

# Calculate predictions using the manual model
test_predictions = sigmoid(np.dot(X_test, weights) + bias)

# Draw the ROC Curve for the manual model
misclassification_rate, F1 = Draw_ROC(y_test, test_predictions, title="ROC Curve for Custom Logistic Regression Model")

print(f"Optimal custom LR classifier has a misclassification rate of: {round(misclassification_rate, 6)}\t and F1 score of: {round(F1, 6)}")


# sklearn pre-made Logistic Regression: ------------------------------------

# Train sklearn logistic regression model (with regularization)
model = LogisticRegression(solver='saga', C=1, max_iter=10000, tol=0.00001)
model.fit(X_train, np.ravel(y_train))

# Calculate predictions using the manual model (Accounting "malignant" as positive class)
test_predictions = model.predict_proba(X_test)[:, [1]]

# Draw the ROC Curve for the manual model
misclassification_rate, F1 = Draw_ROC(y_test, test_predictions, title="ROC Curve for sklearn Logistic Regression Model")

print(f"Optimal sklearn LR classifier has a misclassification rate of: {round(misclassification_rate, 6)}\t and F1 score of: {round(F1, 6)}")


# Custom k-Nearest Neighbours: ------------------------------------

# List to hold the average of the cross-validation losses for every k-nn model
avg_losses = []

k_range = np.arange(1, 18, 2)

for k in k_range:

    # List to hold the cross-validation losses for a k-nn model
    misclassification_rates = []

    for train_indices, valid_indices in KFold_Custom(dataset_size = len(X_train), num_folds = 5):
        
        # Split the training dataset (originally 80% of all dataset) to the training and validation folds
        X_train_fold, X_val_fold = X_train[train_indices], X_train[valid_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[valid_indices]
        
        # Train KNN and predict
        y_pred = knn(X_train_fold, y_train_fold, X_val_fold, k)[:, np.newaxis]

        # Compute the classification metrics
        metrics = Classification_Metrics(y_val_fold, y_pred)

        # Append the 0/1 loss (misclassification rate)
        misclassification_rates.append(metrics["misclassification_rate"])
    
    # Calculate the average of the cross-validation losses for this k-nn model
    avg_loss = np.mean(misclassification_rates)

    # Append the average
    avg_losses.append(avg_loss)

# Plot the average cross-validation losses against (k)
plt.plot(k_range, avg_losses)

# Get optimal k
optimal_k = k_range[np.argmin(avg_losses)]

# Mark the (k) with the minimum average loss
plt.axvline(x = optimal_k, color='red', linestyle='--', linewidth=1)

# Increase axis precision
plt.xticks(np.arange(1, max(k_range)+1, 1))

# Add title and axis labels
plt.title("Average Misclassification Rate of 5-folds Cross-validation\n(Custom-made k-nn model)")
plt.xlabel("k")
plt.ylabel("Average Misclassification Rate")

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Render Image
plt.show()

# Training KNN with the optimal (k) and predict the testing set
y_pred = knn(X_train, y_train, X_test, optimal_k)[:, np.newaxis]

# Computing classification metrics
metrics = Classification_Metrics(y_test, y_pred)

print(f"Optimal custom K-nn model has a k of {optimal_k} and testing misclassification rate of: {round(metrics['misclassification_rate'], 6)}\t and F1 score of: {round(metrics['F1_score'], 6)}")

# sklearn pre-made k-Nearest Neighbours: ------------------------------------

# List to hold the average of the cross-validation losses for every k-nn model
avg_losses = []

k_range = np.arange(1, 18, 2)

for k in k_range:

    # List to hold the cross-validation losses for a k-nn model
    misclassification_rates = []

    # Initialize KFold
    kf = KFold(n_splits=5)

    for train_indices, valid_indices in kf.split(X_train):
        
        # Split the training dataset (originally 80% of all dataset) to the training and validation folds
        X_train_fold, X_val_fold = X_train[train_indices], X_train[valid_indices]
        y_train_fold, y_val_fold = y_train[train_indices], y_train[valid_indices]
        
        # Initialize the KNN classifier with the current k
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Train the classifier
        knn_classifier.fit(X_train_fold, np.ravel(y_train_fold))

        # Make predictions
        y_pred = knn_classifier.predict(X_val_fold)[:, np.newaxis]

        # Compute the classification metrics
        metrics = Classification_Metrics(y_val_fold, y_pred)

        # Append the 0/1 loss (misclassification rate)
        misclassification_rates.append(metrics["misclassification_rate"])
                         
    # Calculate the average of the cross-validation losses for this k-nn model
    avg_loss = np.mean(misclassification_rates)

    # Append the average
    avg_losses.append(avg_loss)

# Plot the average cross-validation losses against (k)
plt.plot(k_range, avg_losses)

# Get optimal k
optimal_k = k_range[np.argmin(avg_losses)]

# Mark the (k) with the minimum average loss
plt.axvline(x = optimal_k, color='red', linestyle='--', linewidth=1)

# Increase axis precision
plt.xticks(np.arange(1, max(k_range)+1, 1))

# Add title and axis labels
plt.title("Average Misclassification Rate of 5-folds Cross-validation\n(Sklearn k-nn model)")
plt.xlabel("k")
plt.ylabel("Average Misclassification Rate")

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Render Image
plt.show()

# Train KNN with the optimal (k) and predict the testing set
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)

# Train the classifier
knn_classifier.fit(X_train, np.ravel(y_train))

# Make predictions
y_pred = knn_classifier.predict(X_test)[:, np.newaxis]

# Computing classification metrics
metrics = Classification_Metrics(y_test, y_pred)

print(f"Optimal Sklearn K-nn model has a k of {optimal_k} and testing misclassification rate of: {round(metrics['misclassification_rate'], 6)}\t and F1 score of: {round(metrics['F1_score'], 6)}")
