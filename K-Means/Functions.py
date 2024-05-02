import numpy as np

def calculate_distances(points, centroids):
        '''
        Function that calculates the Euclidean distance between each point and each centroid.
        '''
        return np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

class KMeans:
    '''
    Manual implementation of K-Means Algorithm
    '''
    def __init__(self, k=3, max_iters=500, init_strategy='random'):

        # Initialize the object parameters
        self.k = k
        self.max_iters = max_iters
        self.init_strategy = init_strategy
        self.centroids = None

    def initialize_centroids(self, points):
        '''
        Method that initializes the centroids based on the chosen 
        1) Randomly pick the centers from the data points.
        2) Pick the centers such that they have a sufficiently large distance between them.
        '''
        if self.init_strategy == "random":
            self.centroids = self.random_initialization(points)
        elif self.init_strategy == "max_distance":
            self.centroids = self.distance_initialization(points)
        else:
            raise ValueError("Unknown initialization strategy.")

    def random_initialization(self, points):
        '''
        Method that randomly picks unique centroids from the data points
        '''
        # Extract unique points to avoid getting two identical centroids
        unique_points = np.unique(points, axis=0)

        # Get K random indices
        indices = np.random.choice(unique_points.shape[0], size=self.k, replace=False)

        # Return the points corresponding to the indices
        return unique_points[indices]

    def distance_initialization(self, points):
        '''
        Method that initializes centroids to have large distances between them
        '''
        # Initialize first centroid randomly
        centroids = [points[np.random.choice(points.shape[0])]]
        
        # To get remaining centroids
        for i in range(1, self.k):

            # Calculate the distance between every point and its nearest centroid
            distances = np.min(calculate_distances(points, np.array(centroids)), axis=0)
            
            # Find the farthest point from it's centroid
            farthest_point = points[np.argmax(distances)]
            
            # Append farthest point as new centroid
            centroids.append(farthest_point)
        
        # Return centroids as a NumPy array.
        return np.array(centroids)

    def assign_clusters(self, points):
        '''
        Method that assigns points to the nearest centroid
        '''
        # Calcualte the distance between each point and each centroid
        distances = calculate_distances(points, np.array(self.centroids))

        # Get the centroids with the least distance to each point
        return np.argmin(distances, axis=0)

    def update_centroids(self, points, clusters):
        ''' 
        Method that updates the clusters centroids
        '''
        # List to store the new centroids
        new_centroids = []

        # For each cluster
        for i in range(self.k):

            # Isolate points of that cluster
            cluster_points = points[clusters == i]

            # If there are points
            if cluster_points.size > 0:
                # Set the mean of those points as new centroid
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # Re-assign the centroid to a random data point
                random_index = np.random.choice(points.shape[0])
                new_centroids.append(points[random_index])
            
        return np.array(new_centroids)

    def fit(self, points):
        '''
        Method that trains the model (fits model to the data)
        '''

        # Initialize the cluster centroids
        self.initialize_centroids(points)
        
        # Variable to indicate whether the training converged and when
        covergerd = None

        for i in range(self.max_iters):

            # Assign the points to their clusters
            clusters = self.assign_clusters(points)

            # Calculate the new clusters centroids
            new_centroids = self.update_centroids(points, clusters)
            
            # Check for convergence (centroids didn't change)
            if np.all(self.centroids == new_centroids):
                print("Converged at iteration:", i)
                covergerd = i
                break
            
            # Set the new centroids
            self.centroids = new_centroids

        return self.centroids, clusters, covergerd 

