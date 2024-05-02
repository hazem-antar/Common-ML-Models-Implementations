import numpy as np

def Generate_Dataset(start, end, n, noise_mean, noise_std):
    ''' 
    Function that generates a dataset with (n) uniformly spaced examples between start and end (inclusive)
    and then adds gaussian noise with specified mean and std.
    returns: A dictionary of features (x) and targets (t)
    '''
    # Pick (n) uniformly spaced feature values between start and end (inclusive)
    x = np.linspace(start, end, n)

    # Noise vector: n samples from a gaussian distribution with mean of noise_mean and std of noise_std
    eps = np.random.normal(noise_mean, noise_std, size=(n))

    # Targets
    t = np.sin(4 * np.pi * x) + eps

    # Return features and target dictionary
    return {"features":x, "targets":t}


def FitLinearModel(Features, Targets, Pseudo_Inverse, lamda_regularization = None):
    ''' 
    Function that fits a linear regression model using the direct
    solution method:
    w = (X^T.X)^(-1).X^T.t   (no ridge regularization)
    or
    w = (X^T.X+(N/2)B)^(-1).X^T.t  (with ridge regularization)
    There is an opetion for using actual or pseudo inverse.
    Inputs: Numpy vectors of processed features and targets
    Returns: Numpy vector of trained model weights
    '''
    # Calculate X^T.X
    J = np.dot(Features.T, Features)

    # If the trainning is with ridge regularization
    if lamda_regularization:
        # Calculate B, where B is a square diagonal matrix of size equal to
        # total number of weights and has 2xlambda in the diagonal elements
        B = np.diag(np.ones(Features.shape[1]) * 2 * lamda_regularization)
        B[0][0] = 0
        
        # Calculate X^T.X + (N/2).B
        # where N is the number of trainning data
        J += (Features.shape[0]/2) * B

    # Calculate the inverse J^-1
    if Pseudo_Inverse: # With Pseudo-Inverse
        inverse = np.linalg.pinv(J)
    else: # With actual inverse
        inverse = np.linalg.inv(J)

    # Calculate the (J^-1.X^T.t)
    weights = np.dot(np.dot(inverse, Features.T), Targets)

    # Return trained weights
    return weights

def CalculateRMSError(Actual, Predicted):
    ''' 
    Function that calculate the Root Mean Square Error (RMSE)
    between the actual and predicted values.
    Input: Numpy vectors of actual and predicted values.
    Returns: A float, the value of the RMSE.
    '''

    # Calculate the square root of the mean of the squared differences
    rmse = np.sqrt(np.mean((Actual - Predicted) ** 2))

    return rmse