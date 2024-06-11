# Utilizes the simulate function to perturb the initial conditions according to either a uniform or
# Gaussian distribution after which the dynamic system is simulated and the results are collected

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

def monte_carlo_sample(initial_condition_dist, dist_bounds, correlation=0, sample_count=100, seed=None):
    """
        Generates samples from either a uniform or gaussian distribution for initial conditions of the states
        @ In, initial_condition_dist, string, normal or uniform
        @ In, dist_bounds, nd.array, (n,2) numpy array of bounds for each initial conditions
        @ In, correlation, float, amount of correlation between each initial condition for gaussian sampling
        @ In, sample_count, int, number of initial condition samples to draw
        @ In, seed, int, random seed to fix samples
        @ Out, monte_samples, np.array, (n, N) where n is system dim and N and sample count
    """
    # Setting seeds
    if seed is not None:
        np.random.seed(seed)

    # Checking for valid inputs
    if initial_condition_dist not in ['normal','uniform']:
        print(f'Invalid input for initial condition distribution')
        exit()

    # Initializing sample set
    monte_samples = np.empty((len(dist_bounds[:,0]), sample_count))

    # When we are using a uniform distribution
    if initial_condition_dist == 'uniform':
        # Sampling within each dimension
        index = 0
        for bounds in dist_bounds:
            monte_samples[index, :] = np.random.uniform(low=bounds[0], high=bounds[1], size=sample_count)
            index += 1
        return monte_samples
    elif initial_condition_dist == 'normal' and correlation == 0:
        # Calculating mean vector and covariance matrix
        mean = np.empty(len(dist_bounds[:,0]))
        variance = np.empty(len(mean))
        index = 0
        for bounds in dist_bounds:
            mean_value = (bounds[1] + bounds[0])/2
            mean[index] = mean_value
            sigma = (1/3)*(mean_value - bounds[0])
            variance[index] = sigma**2
            index += 1
        covariance = np.diag(variance)

        # Let's sample
        return np.transpose(np.random.multivariate_normal(mean, covariance, size=sample_count))
    else:
        print('Still constructing correlated Gaussian, Sorry')
        exit()

if __name__ == '__main__':
    # A few dummy cases for demonstration
    # Uniform first x1 ~ U(-1, 1) and x2 ~ U(-5, 5)
    dist_bounds = np.array([[-1, 1], [-5, 5]])
    monte_samples = monte_carlo_sample('uniform', dist_bounds, sample_count=10)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot the uniform case
    plt.scatter(monte_samples[0,:], monte_samples[1,:])
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    plt.title('Uniform', fontsize=20)
    plt.show()

    # Normal distribution next
    monte_samples = monte_carlo_sample('normal', dist_bounds, sample_count=500)

    # Let's plot the normal case
    plt.scatter(monte_samples[0,:], monte_samples[1,:])
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    plt.title('Normal', fontsize=20)
    plt.show()

    # Testing seeding
    seeded = monte_carlo_sample('uniform', dist_bounds, sample_count=500, seed=42)
    seeded2 = monte_carlo_sample('uniform', dist_bounds, sample_count=500, seed=42)
    test = np.subtract(seeded, seeded2)
    print(test == np.zeros((2, 500)))