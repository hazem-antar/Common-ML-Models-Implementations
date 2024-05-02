from  Functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Seed the pseudo number generator (My ID ends with 0393)
np.random.seed(3093)

# List with the number of clusters (K)'s to try
ks = [2, 3, 10, 20, 40]

# Number of times to train a model using each K
runs = 3

# Experiment with 2 images
for image_path in ["Image1.jpg", "Image2.jpg"]:

    # Load the image
    image = mpimg.imread(image_path)
    original_shape = image.shape
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3) 

    for k in ks:
        print(f"Processing k={k}...")

        # List to fetch a specific initialization method at each run
        init_strategies = ["random", "max_distance"]

        # Create a grid of subplots of size (1 x number of runs)
        fig, axs = plt.subplots(1, runs, figsize=(20, 5)) 

        for run in range(runs):

            # Fetch a specific initialization method 
            init_way = init_strategies[run % len(init_strategies)]

            # Instantiate a K-means model with K clusters
            kmeans = KMeans(k=k, init_strategy=init_way)

            print(f" Run #{run} with initialization: {init_way}")

            # Fit the model on the image pixels
            centroids, clusters, covergerd = kmeans.fit(pixels)
            
            # Reconstruct the image (inference using the trained model)
            segmented_img = centroids[clusters].reshape(original_shape).astype(np.uint8)

            # Calculate the MSE
            mse = np.mean((image - segmented_img) ** 2)

            # Plot the segmented image in its respective subplot
            ax = axs[run]
            ax.imshow(segmented_img)

            # Choose title based on convergence status
            if covergerd:
                title = f'(k={k}, Run #{run}) Converged at {covergerd}\nInit: {init_way}  MSE: {mse:.4f}'
            else:
                title = f'(k={k}, Run #{run}) Didn\'t Converge\nInit: {init_way}  MSE: {mse:.4f}'

            ax.set_title(title)

            # Hide axes ticks
            ax.axis('off') 


        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()