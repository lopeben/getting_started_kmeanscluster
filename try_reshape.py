import numpy as np

# Define your 1D numpy array
Arr = np.array([1, 2, 3, 4, 5, 6])

n_samples = 2  # number of samples
n_features = 3  # number of features

# Reshape the 1D array to 2D array with 'n_samples' rows and 'n_features' columns
Arr_2D = Arr.reshape(n_samples, n_features)

print(Arr_2D)
