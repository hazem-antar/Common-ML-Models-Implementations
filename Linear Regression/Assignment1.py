import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from Functions import *

# Seed the pseudo number generator (My ID ends with 0393)
np.random.seed(3093)

# Trainning Set:
TrainingSet = Generate_Dataset(0, 1, 10, 0, 0.09)

# Validation Set:
ValidationSet = Generate_Dataset(0, 1, 100, 0, 0.09)

# Testing Set:
TestingSet = Generate_Dataset(0, 1, 100, 0, 0.09)


### Question 1, 2 ###

# Lists to store the calculated losses
TrainningLosses = []
ValidationLosses = []

# Setting up a grid of 5*2 subplots
predictors_fig, predictors_axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 11), sharex=True)

# Flatten the predictors_axes list to be one dimentional
predictors_axes = predictors_axes.flatten()

# Generating and plotting f_M(x) for 10 regression models
for M in range(10):

    # Phase 1: Trainning  -----------------------

    # Transforming the features to include (M) polynomial terms (including bias)
    poly = PolynomialFeatures(degree=M, include_bias=True)

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
    TrinningTargets = TrainingSet["targets"].reshape(-1, 1)
    
    # Transforming the features to include the powers of the original features
    PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

    # Fit a Linear Regression model (using pseudo-inverse for numerical stability)
    weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True)

    # Calculate the predictions of training samples
    Predicted_Trianning = np.dot(PolyTrainningFeatures, weights)
    
    # Calculate the trainning loss
    trainning_loss = CalculateRMSError(TrinningTargets, Predicted_Trianning)

    # Appending the trainning loss
    TrainningLosses.append(trainning_loss)

    # Phase 2: Validation -----------------------

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    ValidationFeatures = ValidationSet["features"].reshape(-1, 1)
    ValidationTargets = ValidationSet["targets"].reshape(-1, 1)

    # Transforming the validation features to correct ploynomial form
    PolyValidationFeatures = poly.fit_transform(ValidationFeatures) 

    # Calculate the predictions of validation samples
    Predicted_Validation = np.dot(PolyValidationFeatures, weights)

    # Calculate the validation loss
    validation_loss = CalculateRMSError(ValidationTargets, Predicted_Validation)

    # Appending the validation loss
    ValidationLosses.append(validation_loss)

    # Phase 3: Plotting -----------------------

    # Defining the x-values to plot the functions (increasing the steps increases the plotting accuracy)
    x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
   
    # Transforming the x-values to correct ploynomial form
    PolyX_range = poly.fit_transform(x_range) 

    # Calculating f_M(x)
    y_estimator = np.dot(PolyX_range, weights)

    # Calculating the f_true(x)
    y_true = np.sin(4 * np.pi * x_range)

    # Select the subplot axis to plot on
    ax = predictors_axes[M]

    # Plot the true and estimator curves
    ax.plot(x_range, y_true, color='green', linewidth=0.75)
    ax.plot(x_range, y_estimator, color='blue', linewidth=0.75, linestyle='--')

    # Scatter the trainning and validation samples
    ax.scatter(TrainingSet['features'], TrainingSet['targets'], color='red', marker='o', s=3)
    ax.scatter(ValidationSet['features'], ValidationSet['targets'], color='orange', marker='x', s=3)
    
    # Plot title labels, and legend
    ax.set_title(f'Polynomial Regression of Degree {M}', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

# Adjust the spacing between the subplots
predictors_fig.subplots_adjust(hspace=0.5, top=0.96)

# Create a single legend for the entire figure
labels = ['True Function', 'Prediction', 'Training Data', 'Validation Data']
lines = [plt.Line2D([0], [0], color='green'),
         plt.Line2D([0], [0], color='blue', linestyle='--'),
         plt.Line2D([0], [0], color='red', marker='o', linestyle=''),
         plt.Line2D([0], [0], color='orange', marker='x', linestyle='')]
predictors_fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.1, 0.5))

# Save the figure of the 10 regression models
plt.savefig('polynomial_regression_predictors.png', format='png', bbox_inches='tight')

# Plot a graph comparing the trainning and validation losses of the predictors
losses_fig = plt.figure(figsize=(10, 6))
plt.plot(np.arange(10), TrainningLosses, color='green', label = 'Trainning Loss', linewidth=1)
plt.plot(np.arange(10), ValidationLosses, color='blue', label = 'Validation Loss', linewidth=1)  

# Drawing a the minimum validation loss
plt.axvline(x=np.argmin(ValidationLosses), color='red', linestyle='--', linewidth=1)
plt.text(x=1.1, y=0.1, s=f" Minimum validation loss = {round(np.min(ValidationLosses), 3)} (M = {np.argmin(ValidationLosses)})")

# Add title, labels and legend
plt.title("Comparing RMSEs of Different Predictor Capacities (M)", fontsize=13) 
plt.xlabel('Model Capacity (M)')
plt.ylabel('RMSE')
plt.legend()

# Save the losses comparison
plt.savefig('predictor_losses.png', format='png', bbox_inches='tight')

#--------------------------------------------------
### Question 3 ###

# Lists to store the calculated losses
TrainningLosses = []
ValidationLosses = []

# List to store the generated lambda value
lambdas = []

# Trainning muliple regression models with different lambdas
for i in range(30):

    # Calculating a specific lambda
    lamda = 5.0**(-i)
    lambdas.append(lamda)

    # Phase 1: Trainning  -----------------------

    # Transforming the features to include (M) polynomial terms (including bias)
    poly = PolynomialFeatures(degree=9, include_bias=True)

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
    TrinningTargets = TrainingSet["targets"].reshape(-1, 1)
    
    # Transforming the features to include the powers of the original features
    PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

    # Standerdizing the trainnig features 
    scaler = StandardScaler()
    PolyTrainningFeatures = scaler.fit_transform(PolyTrainningFeatures)

    # Fit a Linear Regression model with ridge regularization and pseudo-inverse
    weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True, lamda_regularization = lamda)

    # Calculate the predictions of training samples
    Predicted_Trianning = np.dot(PolyTrainningFeatures, weights)
    
    # Calculate the trainning loss
    trainning_loss = CalculateRMSError(TrinningTargets, Predicted_Trianning)

    # Appending the trainning loss
    TrainningLosses.append(trainning_loss)

    # Phase 2: Validation -----------------------

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    ValidationFeatures = ValidationSet["features"].reshape(-1, 1)
    ValidationTargets = ValidationSet["targets"].reshape(-1, 1)

    # Transforming the validation features to correct ploynomial form
    PolyValidationFeatures = poly.fit_transform(ValidationFeatures) 

    # Standerdizing the validation features 
    PolyValidationFeatures = scaler.transform(PolyValidationFeatures)

    # Calculate the predictions of validation samples
    Predicted_Validation = np.dot(PolyValidationFeatures, weights)

    # Calculate the validation loss
    validation_loss = CalculateRMSError(ValidationTargets, Predicted_Validation)

    # Appending the validation loss
    ValidationLosses.append(validation_loss)

# Get a log_10 version of lambda for ease of plotting
log_lambda = np.log10(lambdas)

# Plot a graph comparing the trainning and validation losses of the predictors
losses_fig = plt.figure(figsize=(10, 6))
plt.plot(log_lambda, TrainningLosses, color='green', label = 'Trainning Loss', linewidth=1)
plt.plot(log_lambda, ValidationLosses, color='blue', label = 'Validation Loss', linewidth=1)  

# Defining a the minimum validation loss
plt.axvline(x=log_lambda[np.argmin(ValidationLosses)], color='red', linestyle='--', linewidth=1)
plt.text(x=-17, y=0.1, s=f" Minimum validation loss = {round(np.min(ValidationLosses), 3)} (λ = {lambdas[np.argmin(ValidationLosses)]})")

# Add title, labels and legend
plt.title("Comparing RMSEs of 9-th Degree Predictors with different λ", fontsize=13) 
plt.xlabel('log_10(λ)')
plt.ylabel('RMSE')
plt.legend()

# Save the losses comparison
plt.savefig('predictor_losses_lambdas.png', format='png', bbox_inches='tight')


### Queston 4 ###

# Setting up a grid of 2*1 subplots
predictors_fig, predictors_axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 2), sharex=True)

# Flatten the predictors_axes list to be one dimentional
predictors_axes = predictors_axes.flatten()

# Trainning and plotting f_M(x) for 2 models with different lambdas
for i, lamda in enumerate([2.048e-08 , 1]):

    # Phase 1: Trainning  -----------------------

    # Transforming the features to include (M) polynomial terms (including bias)
    poly = PolynomialFeatures(degree=9, include_bias=True)

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
    TrinningTargets = TrainingSet["targets"].reshape(-1, 1)
    
    # Transforming the features to include the powers of the original features
    PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

    # Standerdizing the trainnig features 
    scaler = StandardScaler()
    PolyTrainningFeatures = scaler.fit_transform(PolyTrainningFeatures)

    # Fit a Linear Regression model with ridge regularization and pseudo-inverse
    weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True, lamda_regularization = lamda)

    # Phase 3: Plotting -----------------------

    # Defining the x-values to plot the functions (increasing the steps increases the plotting accuracy)
    x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
   
    # Transforming the x-values to correct ploynomial form
    PolyX_range = poly.fit_transform(x_range) 

    # Standerdizing the polynomial features 
    PolyX_range = scaler.transform(PolyX_range)

    # Calculating f_M(x)
    y_estimator = np.dot(PolyX_range, weights)

    # Calculating the f_true(x)
    y_true = np.sin(4 * np.pi * x_range)

    # Select the subplot axis to plot on
    ax = predictors_axes[i]

    # Plot the true and estimator curves
    ax.plot(x_range, y_true, color='green', linewidth=0.75)
    ax.plot(x_range, y_estimator, color='blue', linewidth=0.75, linestyle='--')

    # Scatter the trainning and validation samples
    ax.scatter(TrainingSet['features'], TrainingSet['targets'], color='red', marker='o', s=3)
    ax.scatter(ValidationSet['features'], ValidationSet['targets'], color='orange', marker='x', s=3)
    
    # Plot title labels, and legend
    ax.set_title(f'9-th Degree Polynomial with λ = {lamda}', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

# Adjust the spacing between the subplots
predictors_fig.subplots_adjust(hspace=0.5, top=0.96)

# Create a single legend for the entire figure
labels = ['True Function', 'Prediction', 'Training Data', 'Validation Data']
lines = [plt.Line2D([0], [0], color='green'),
         plt.Line2D([0], [0], color='blue', linestyle='--'),
         plt.Line2D([0], [0], color='red', marker='o', linestyle=''),
         plt.Line2D([0], [0], color='orange', marker='x', linestyle='')]
predictors_fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.05, 0.7))

# Save the figure of the 10 regression models
plt.savefig('Comparing_two_lambdas.png', format='png', bbox_inches='tight')


### Question 5 ###

# Trainning the best model 

# Transforming the features to include (M) polynomial terms (including bias)
poly = PolynomialFeatures(degree=9, include_bias=True)

# Reshape the feature vector adding extra dimention to allow the use of fit_transfom
TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
TrinningTargets = TrainingSet["targets"].reshape(-1, 1)

# Transforming the features to include the powers of the original features
PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

# Standerdizing the trainnig features 
scaler = StandardScaler()
PolyTrainningFeatures = scaler.fit_transform(PolyTrainningFeatures)

# Fit a Linear Regression model with ridge regularization and pseudo-inverse
weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True, lamda_regularization = 2.048e-08)

# Reshape the feature vector adding extra dimention to allow the use of fit_transfom
TestFeatures = TestingSet["features"].reshape(-1, 1)
TestTargets = TestingSet["targets"].reshape(-1, 1)

# Transforming the test features to correct ploynomial form
PolyTestFeatures = poly.fit_transform(TestFeatures) 

# Standerdizing the test features 
PolyTestFeatures = scaler.transform(PolyTestFeatures)

# Calculate the predictions of validation samples
Predicted_Testing = np.dot(PolyTestFeatures, weights)

# Calculate the testing loss
testing_loss = CalculateRMSError(TestTargets, Predicted_Testing)

print("Testing loss of best model:", testing_loss)

### Queston 6 ###

# Setting up a large figure for all predictors
combined_predictors= plt.figure(figsize=(20, 10))

# Generating and plotting f_M(x) for 10 regression models without regularization
for M in range(10):

    # Phase 1: Trainning  -----------------------

    # Transforming the features to include (M) polynomial terms (including bias)
    poly = PolynomialFeatures(degree=M, include_bias=True)

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
    TrinningTargets = TrainingSet["targets"].reshape(-1, 1)
    
    # Transforming the features to include the powers of the original features
    PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

    # Fit a Linear Regression model (using pseudo-inverse for numerical stability)
    weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True)

    # Phase 2: Plotting -----------------------

    # Defining the x-values to plot the functions (increasing the steps increases the plotting accuracy)
    x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
   
    # Transforming the x-values to correct ploynomial form
    PolyX_range = poly.fit_transform(x_range) 

    # Calculating f_M(x)
    y_estimator = np.dot(PolyX_range, weights)

    # Plot the estimators
    plt.plot(x_range, y_estimator, linewidth=0.75, linestyle='--', label= f'{M}-th Degree')

# Trainning and plotting f_M(x) for 2 models with different lambdas
for lamda in [2.048e-08 , 1]:

    # Phase 1: Trainning  -----------------------

    # Transforming the features to include (M) polynomial terms (including bias)
    poly = PolynomialFeatures(degree=9, include_bias=True)

    # Reshape the feature vector adding extra dimention to allow the use of fit_transfom
    TrainningFeatures = TrainingSet["features"].reshape(-1, 1)
    TrinningTargets = TrainingSet["targets"].reshape(-1, 1)
    
    # Transforming the features to include the powers of the original features
    PolyTrainningFeatures = poly.fit_transform(TrainningFeatures)

    # Standerdizing the trainnig features 
    scaler = StandardScaler()
    PolyTrainningFeatures = scaler.fit_transform(PolyTrainningFeatures)

    # Fit a Linear Regression model with ridge regularization and pseudo-inverse
    weights = FitLinearModel(PolyTrainningFeatures, TrinningTargets, Pseudo_Inverse = True, lamda_regularization = lamda)

    # Phase 2: Plotting -----------------------

    # Defining the x-values to plot the functions (increasing the steps increases the plotting accuracy)
    x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
   
    # Transforming the x-values to correct ploynomial form
    PolyX_range = poly.fit_transform(x_range) 

    # Standerdizing the polynomial features 
    PolyX_range = scaler.transform(PolyX_range)

    # Calculating f_M(x)
    y_estimator = np.dot(PolyX_range, weights)

    # Plot the estimator curves
    plt.plot(x_range, y_estimator, linewidth=0.75, linestyle='--', label= f'9-th Degree\nλ = {lamda}')

# Calculating the f_true(x)
y_true = np.sin(4 * np.pi * x_range)

# Plot the true curves
plt.plot(x_range, y_true, color='green', linewidth=0.75, label = 'True Function')

# Scatter the trainning and validation samples
plt.scatter(TrainingSet['features'], TrainingSet['targets'], color='red', marker='o', s=5, label = 'Training Data')
plt.scatter(ValidationSet['features'], ValidationSet['targets'], color='orange', marker='x', s=5, label = 'Validation Data')

# Calculating the f_true(validation features)
y_true_validation = np.sin(4 * np.pi * ValidationSet["features"])

# Calculate the RMSE between the targets and f_true(validation features)
f_true_loss = round(CalculateRMSError(ValidationSet["targets"], y_true_validation), 4)

# Plot title labels, and legend
plt.title(f'Comparing all twelve predictors against true function\n(RMSE between (t) and f_true(x) for validation data = {f_true_loss})', fontsize=14)
plt.xlabel('x')
plt.ylabel('f(x)')

plt.legend()

# Save the figure of the 12 regression models
plt.savefig('polynomial_regression_12_predictors.png', format='png', bbox_inches='tight')
