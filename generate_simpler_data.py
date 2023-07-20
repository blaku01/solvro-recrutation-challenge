import numpy as np

def generate_parabolic_trajectory_data(N_samples, trajectory_length=300, a=0.05, b=0.0, c=0.0, max_noise=0.5):
    """
    Generate synthetic 3-dimensional data of shape (N_samples, trajectory_length, 2) with parabolic trajectories.

    Parameters:
        N_samples (int): Number of samples (trajectories) to generate.
        trajectory_length (int): Length of each trajectory (number of time steps). Default is 300.
        a (float): Coefficient of the x^2 term in the parabolic equation. Controls the curvature of the parabola.
        b (float): Coefficient of the x term in the parabolic equation. Controls the linear term.
        c (float): Constant term in the parabolic equation. Controls the vertical shift of the parabola.
        max_noise (float): Maximum noise added to each y-coordinate to create variation in the trajectories.

    Returns:
        ndarray: Array of shape (N_samples, trajectory_length, 2) containing synthetic 2D parabolic trajectories.
    """
    data = np.zeros((N_samples, trajectory_length, 2))
    x_values = np.linspace(-10, 10, trajectory_length)

    for i in range(N_samples):
        y_values = a * x_values**2 + b * x_values + c
        noise = np.random.uniform(-max_noise, max_noise, size=trajectory_length)
        y_values += noise

        data[i, :, 0] = x_values
        data[i, :, 1] = y_values

    return data

# Generate 3-dimensional data of shape (N_samples, 300, 2) with parabolic trajectories
N_samples = 1000  # You can change this number to generate more or fewer samples
data_parabolic = generate_parabolic_trajectory_data(N_samples)

print(data_parabolic.shape)  # Output: (N_samples, 300, 2)

import numpy as np

def generate_linear_trajectory_data(N_samples, trajectory_length=300, slope=0.2, intercept=0.0, max_noise=0.5):
    """
    Generate synthetic 3-dimensional data of shape (N_samples, trajectory_length, 2) with linear trajectories.

    Parameters:
        N_samples (int): Number of samples (trajectories) to generate.
        trajectory_length (int): Length of each trajectory (number of time steps). Default is 300.
        slope (float): Slope of the straight line.
        intercept (float): Y-intercept of the straight line.
        max_noise (float): Maximum noise added to each y-coordinate to create variation in the trajectories.

    Returns:
        ndarray: Array of shape (N_samples, trajectory_length, 2) containing synthetic 2D linear trajectories.
    """
    data = np.zeros((N_samples, trajectory_length, 2))
    x_values = np.linspace(-10, 10, trajectory_length)

    for i in range(N_samples):
        y_values = slope * x_values + intercept
        noise = np.random.uniform(-max_noise, max_noise, size=trajectory_length)
        y_values += noise

        data[i, :, 0] = x_values
        data[i, :, 1] = y_values

    return data

# Generate 3-dimensional data of shape (N_samples, 300, 2) with linear trajectories
N_samples = 10000  # You can change this number to generate more or fewer samples
data_linear = generate_linear_trajectory_data(N_samples)

print(data_linear.shape)  # Output: (N_samples, 300, 2)


import numpy as np

# Assuming you have already generated data_linear and data_parabolic
# If not, please run the code to generate those datasets first.

# Create class labels (0 for linear, 1 for parabolic)
num_linear_samples = data_linear.shape[0]
num_parabolic_samples = data_parabolic.shape[0]
y_linear = np.zeros(num_linear_samples)
y_parabolic = np.ones(num_parabolic_samples)

# Combine the data and labels
X_data = np.concatenate((data_linear, data_parabolic), axis=0)
y_data = np.concatenate((y_linear, y_parabolic), axis=0)

# Shuffle the data
shuffle_indices = np.random.permutation(len(X_data))
X_data_shuffled = X_data[shuffle_indices]
y_data_shuffled = y_data[shuffle_indices]

# Split the data into training, validation, and test sets (60% train, 20% validation, 20% test)
train_ratio = 0.6
val_ratio = 0.2
train_index = int(len(X_data_shuffled) * train_ratio)
val_index = int(len(X_data_shuffled) * (train_ratio + val_ratio))

X_train = X_data_shuffled[:train_index]
X_val = X_data_shuffled[train_index:val_index]
X_test = X_data_shuffled[val_index:]

y_train = y_data_shuffled[:train_index]
y_val = y_data_shuffled[train_index:val_index]
y_test = y_data_shuffled[val_index:]

# Save the datasets as numpy arrays
np.save('simple_sample_data/X_train_filtered.npy', X_train)
np.save('simple_sample_data/X_val.npy', X_val)
np.save('simple_sample_data/X_test.npy', X_test)
np.save('simple_sample_data/y_train.npy', y_train)
np.save('simple_sample_data/y_val.npy', y_val)
np.save('simple_sample_data/y_test.npy', y_test)
