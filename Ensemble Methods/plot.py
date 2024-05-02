import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv("ensemble_losses.csv")

# Step 2: Extract data
max_predictors = df['Number of Predictors']
BaggingClassifierlosses = df['Bagging Loss']
RandomForestlosses = df['Random Forest Loss']
AdaboostStumpsClassifier_losses = df['AdaBoost Stumps Loss']
AdaboostTree10Classifier_losses = df['AdaBoost Tree 10 Leaves Loss']
AdaboostTreeClassifier_losses = df['AdaBoost Unrestricted Tree Loss']
best_DTree_loss = df['Decision Tree Loss'][0]

# Mapping of column names to labels for plotting
losses_to_plot = {
    'Bagging Loss': "Bagging with Decision Trees",
    'Random Forest Loss': "Random Forest",
    'AdaBoost Stumps Loss': "AdaBoost with Decision Stumps",
    'AdaBoost Tree 10 Leaves Loss': "AdaBoost with Trees (max 10 leaves)",
    'AdaBoost Unrestricted Tree Loss': "AdaBoost with Unrestricted Trees"
}

# Loop through the dictionary and create a plot for each
for column, label in losses_to_plot.items():
    plt.figure(figsize=(8, 5))
    plt.plot(max_predictors, df[column], label=label)
    plt.xlabel("Number of Predictors")
    plt.ylabel("Misclassification Rate (0/1 Loss)")
    plt.title(f"{label} Performance")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


# Step 4: Combined plot
plt.figure(figsize=(10, 6))
plt.plot(max_predictors, BaggingClassifierlosses, label="Bagging with Decision Trees")
plt.plot(max_predictors, RandomForestlosses, label="Random Forest")
plt.plot(max_predictors, AdaboostStumpsClassifier_losses, label="AdaBoost with Decision Stumps")
plt.plot(max_predictors, AdaboostTree10Classifier_losses, label="AdaBoost with Trees (max 10 leaves)")
plt.plot(max_predictors, AdaboostTreeClassifier_losses, label="AdaBoost with Unrestricted Trees")
plt.axhline(y=best_DTree_loss, color='r', linestyle='--', label="Best Decision Tree Loss")
plt.xticks(np.arange(50, 2501, 200))
plt.xlabel("Number of Predictors")
plt.ylabel("Misclassification Rate (0/1 Loss)")
plt.title("Classifier Performance Comparison")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
