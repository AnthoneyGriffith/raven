# Uses initial conditions drawn from monte carlo sampling and simulates a dynamic system for each
# The results are then conglogmerated for the purposes of performance evaluation

import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import monte_carlo_sample
from simulate import simulate

def multi_simulation(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds, sample_count):
    """
        Uses various initial conditions to simulate a dynamic system and return the conglomerate results
        @ In, closed_loop_dynamics, python function f(t,x), takes in time and state and returns dynamics
        @ In, tsteps, np.array, set of times to return states over from integration
        @ In, initial_condition_dist, string, normal or uniform
        @ In, dist_bounds, nd.array, (n,2) numpy array of bounds for each initial conditions
        @ In, sample_count, int, number of initial condition samples to draw
        @ Out, simulation_dict, dict, Sample_#:odesol object
    """
    # Sampling initial conditions of closed-loop dynamic system
    monte_initials = monte_carlo_sample(initial_condition_dist, dist_bounds, sample_count=sample_count)
    
    # Run each simulation
    sample = 1
    simulation_dict = {}
    for initial_con in np.transpose(monte_initials):
        simulation_dict.update({f'Sample_{sample}':simulate(closed_loop_dynamics, initial_con, tsteps)})
        sample += 1
    
    return simulation_dict

if __name__ == '__main__':
    print('Testing multi simulator for demonstration')

    # Using a harmonic oscillator to test functionality
    def harmonic_osc(t, x, u=0, omega_0=10, zeta=0.15, m=2):
        """
            Harmonic oscillator dynamics
        """
        xdot = np.empty(len(x))
        xdot[0] = x[1]
        xdot[1] = -1*np.square(omega_0)*x[0] - 2*zeta*omega_0*x[1] + (1/m)*u
        return xdot
    
    # Testing integrator and plotting results
    closed_loop_dynamics = lambda t, y: harmonic_osc(t, x=y)
    dist_bounds = np.array([[-2, 2], [-3, 3]])
    sample_count = 50
    initial_condition_dist = 'uniform'
    tsteps = np.linspace(0, 5, 200)

    sols = multi_simulation(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds, sample_count)

    # Let's plot the multi-solution
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    for samp, sol in sols.items():
       ax1.plot(sol['t'], sol['y'][0,:], color='blue', linewidth=1.25, alpha=0.5)
       ax2.plot(sol['t'], sol['y'][1,:], color='red', linewidth=1.25, alpha=0.5)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=16)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=16)
    ax2.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()
    fig.suptitle('Oscillator Multi-Simulation (Uniform)', fontsize=20)
    plt.show()

    # Let's try using a normal distribution now
    initial_condition_dist = 'normal'
    sols = multi_simulation(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds, sample_count)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    for samp, sol in sols.items():
       ax1.plot(sol['t'], sol['y'][0,:], color='blue', linewidth=1.25, alpha=0.5)
       ax2.plot(sol['t'], sol['y'][1,:], color='red', linewidth=1.25, alpha=0.5)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=16)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=16)
    ax2.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()
    fig.suptitle('Oscillator Multi-Simulation (Normal)', fontsize=20)
    plt.show()