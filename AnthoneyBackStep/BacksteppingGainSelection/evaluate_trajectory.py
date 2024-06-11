# Calculates performance metrics for a given trajectory like settling time, overshoot, and crossings
import numpy as np
from simulate import simulate
import matplotlib.pyplot as plt 

def evaluate_trajectory(trajectory_info, settling_time_tol=1e-3, rise_time_ratio=0.1, desired_state=None):
    """
        Calculates estimates of settling time, overshoot, and cross-over count
        @ In, trajectory_info, ode solution object, information from simulation
        @ In, settling_time_tol, the maximum allowed difference between state and desired to count as settled
        @ In, rise_time_ratio, float, the ratio of the starting state difference necessary to consider the signal risen
        @ In, desired_state, np.array, desired states to converge to
        @ Out, Ts, np.array, approximate settling time for each state
        @ Out, Tr, np.array, approximate rise time for each state
        @ Out, OS, np.array, approximate overshoot amount in transient response for each state
        @ Out, Osc, np.array, number of crossings of transient state with desired for each state
    """
    # Retrieve state and time trajectories
    X = trajectory_info['y']
    T = trajectory_info['t']

    # Check if a desired_state has been provided
    if desired_state is None:
        desired_state = np.zeros(len(X[:,0]))

    # Initializing outputs
    Ts = np.empty(len(desired_state))
    Tr = np.empty(len(desired_state))
    OS = np.empty(len(desired_state))
    Osc = np.empty(len(desired_state))

    # Iterate through each state's trajectory
    index = 0
    for state in X:
        x_d = desired_state[index]
        # Let's start with settling time
        Ts_state = T[-1]
        settled = False
        for t_ind, x_t in enumerate(state):
            dist = np.abs(np.subtract(x_t, x_d))
            # If dist is less than tol we might be settled
            if dist <= settling_time_tol and settled == False:
                Ts_state = T[t_ind]
                settled = True
            # if dist is greater than, want to make sure we mark as not settled anymore
            elif dist > settling_time_tol and settled == True:
                Ts_state = T[-1]
                settled = False
            else:
                continue
        Ts[index] = Ts_state

        # Estimating rise time of the the transient response
        rise_dist = rise_time_ratio*np.abs(np.subtract(x_d, state[0]))
        for t_ind, x_t in enumerate(state):
            dist = np.abs(np.subtract(x_t, x_d))
            # Counting first approach to x_d as rise
            if dist <= rise_dist:
                Tr[index] = T[t_ind]
                break

        # Time to estimate overshoot ignoring first approach to desired state, assuming crossing
        if state[0] <= x_d:
            approach = 'increase'
        else:
            approach = 'decrease'
        cross_ind = None
        # Finding where we cross first
        for t_ind, x_t in enumerate(state):
            # Case 1
            if approach == 'increase' and x_t > x_d:
                cross_ind = t_ind
                break

            # Case 2
            elif approach == 'decrease' and x_t < x_d:
                cross_ind = t_ind
                break

            # No crossover
            else:
                continue
        # Overshoot estimation
        if cross_ind is not None:
            OS[index] = np.max(np.abs(np.subtract(state[cross_ind:], x_d)))
        else:
            OS[index] = 0

        # Tracking number of crossings with desired state
        if approach == 'increase':
            old_sgn = 0
        else:
            old_sgn = 1
        cross_count = 0
        for x_t in state[1:]:
            # Checking sgn (below or above desired)
            if x_t <= x_d - settling_time_tol:
                sgn = 0
            elif x_t >= x_d + settling_time_tol:
                sgn = 1
            else: 
                continue
            cross_count += np.abs(np.subtract(sgn, old_sgn))
            old_sgn = sgn
        Osc[index] = cross_count
        index += 1

    return Ts, Tr, OS, Osc   

if __name__ == '__main__':
    print('Testing trajectory evaluation for demonstration')

    # Using a harmonic oscillator to test functionality
    def harmonic_osc(t, x, u=0, omega_0=10, zeta=0.15, m=1):
        """
            Harmonic oscillator dynamics
        """
        xdot = np.empty(len(x))
        xdot[0] = x[1]
        xdot[1] = -1*np.square(omega_0)*x[0] - 2*zeta*omega_0*x[1] + (1/m)*u
        return xdot
    
    # Testing integrator and plotting results
    closed_loop_dynamics = lambda t, y: harmonic_osc(t, x=y)
    initial_conditions = np.array([1.5, -0.5])
    tsteps = np.linspace(0, 10, 10000)

    solution = simulate(closed_loop_dynamics, initial_conditions, tsteps)
    Ts, Tr, OS, Osc = evaluate_trajectory(solution)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot trajectory first then print the performance metrics
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(solution['t'], solution['y'][0,:], color='blue', linewidth=1.5)
    ax2.plot(solution['t'], solution['y'][1,:], color='red', linewidth=1.5)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=16)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=16)
    ax2.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()

    print(f'Settling time for Position: {Ts[0]}')
    print(f'Rise time for Position: {Tr[0]}')
    print(f'Overshoot estimate for Position: {OS[0]}')
    print(f'Crossing count for Position: {Osc[0]}')
    print(f'Settling time for Velocity: {Ts[1]}')
    print(f'Rise time for Velocity: {Tr[1]}')
    print(f'Overshoot estimate for Velocity: {OS[1]}')
    print(f'Crossing count for Velocity: {Osc[1]}')

    plt.show()