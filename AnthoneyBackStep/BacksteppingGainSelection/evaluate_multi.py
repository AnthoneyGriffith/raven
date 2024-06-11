# Script for evaluating each trajectory and then averaging over the results to approximate expectations
import numpy as np
import matplotlib.pyplot as plt
from generate_control import generate_control, simple_open_loop, feedback_control
from multi_simulation import multi_simulation
from evaluate_trajectory import evaluate_trajectory
from evaluate_control import evaluate_control

def evaluate_multi(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds,
                   sample_count=100, settling_time_tol=1e-3, rise_time_ratio=0.1, desired_state=None,
                   control_law=None):
    """
        Simulates many realizations of closed loop dynamics and evaluates performance metrics
        @ In, closed_loop_dynamics, python function f(t, x, u(t,x)), takes in time and state and returns dynamics
        @ In, tsteps, np.array, set of times to return states over from integration
        @ In, initial_condition_dist, string, normal or uniform
        @ In, dist_bounds, nd.array, (n,2) numpy array of bounds for each initial conditions
        @ In, sample_count, int, number of initial condition samples to draw
        @ In, settling_time_tol, the maximum allowed difference between state and desired to count as settled
        @ In, rise_time_ratio, float, the ratio of the starting state difference necessary to consider the signal risen
        @ In, desired_state, np.array, desired states to converge to
        @ In, control_law, python function u(t, x), takes in time and state and returns control value
        @ Out, multi_traj, dict, set of trajectory outputs from multiple simulations
        @ Out, multi_ctrl, dict, set of control behaviors form multiple simulations
        @ Out, E_Ts, np.array, approximate expected value of settling time
        @ Out, E_Tr, np.array, approximate expected value of rising time
        @ Out, E_OS, np.array, approximate expected value of Overshoot
        @ Out, E_Osc, np.array, approximate expected value of cross count
        @ Out, E_Up, np.array, approximate expected peak control value
        @ Out, E_Ui, np.array, approximate expected integral of the absolute value of control
    """
    # In case no control law is provided
    if control_law is None:
        control_law = lambda t, y: np.zeros(1)

    # Run multi-simulation
    multi_traj = multi_simulation(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds, sample_count=sample_count)
    
    # Retrieving dummy control to calculate output dimensions
    dummy_ctrl = generate_control(control_law, solution_dict=multi_traj[list(multi_traj)[0]])
    
    # Initializing outputs
    E_Ts = np.zeros(len(dist_bounds[:,0]))
    E_Tr = np.zeros(len(dist_bounds[:,0]))
    E_OS = np.zeros(len(dist_bounds[:,0]))
    E_Osc = np.zeros(len(dist_bounds[:,0]))
    E_Up = np.zeros(len(dummy_ctrl[:,0]))
    E_Ui = np.zeros(len(dummy_ctrl[:,0]))

    # Evaluating each trajectory and storing control results
    multi_ctrl = {}
    for samp_name in list(multi_traj):
        Ts, Tr, OS, Osc = evaluate_trajectory(multi_traj[samp_name], settling_time_tol=settling_time_tol, rise_time_ratio=rise_time_ratio, desired_state=desired_state)
        Up, Ui, control_vec = evaluate_control(control_law, multi_traj[samp_name])
        multi_ctrl[samp_name] = control_vec
        E_Ts = np.add(E_Ts, Ts)
        E_Tr = np.add(E_Tr, Tr)
        E_OS = np.add(E_OS, OS)
        E_Osc = np.add(E_Osc, Osc)
        E_Up = np.add(E_Up, Up)
        E_Ui = np.add(E_Ui, Ui)

    E_Ts = np.divide(E_Ts, sample_count)
    E_Tr = np.divide(E_Tr, sample_count)
    E_OS = np.divide(E_OS, sample_count)
    E_Osc = np.divide(E_Osc, sample_count)
    E_Up = np.divide(E_Up, sample_count)
    E_Ui = np.divide(E_Ui, sample_count)

    return multi_traj, multi_ctrl, E_Ts, E_Tr, E_OS, E_Osc, E_Up, E_Ui

if __name__ == '__main__':
    print('Testing multi evaluation for demonstration')

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
    # closed_loop_dynamics = lambda t, y: harmonic_osc(t, x=y)
    control_law = lambda t, y: feedback_control(t, y)
    closed_loop_dynamics = lambda t, y : simple_open_loop(t, X=y, U=control_law(t, y))
    dist_bounds = np.array([[-2, 2], [-3, 3]])
    sample_count = 50
    initial_condition_dist = 'uniform'
    tsteps = np.linspace(0, 10, 10000)

    sols, ctrls, E_Ts, E_Tr, E_OS, E_Osc, E_Up, E_Ui = evaluate_multi(closed_loop_dynamics, tsteps, initial_condition_dist, dist_bounds,
                                                                      settling_time_tol=0.05, sample_count=sample_count, control_law=control_law)

    # Let's plot the multi-solution
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot trajectories
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    # Plotting each trajectory and control
    for samp_name, sol in sols.items():
        ax1.plot(sol['t'], sol['y'][0,:], color='blue', linewidth=1.25, alpha=0.5)
        ax2.plot(sol['t'], sol['y'][1,:], color='red', linewidth=1.25, alpha=0.5)
        ax3.plot(sol['t'], ctrls[samp_name][0,:], color='green', linewidth=1.25, alpha=0.5)

    ax1.set_ylabel('$x_1$ (Position)', fontsize=16)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=16)
    ax3.set_ylabel('$u$ (Control)', fontsize=16)
    ax3.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    print(f'Expected settling time for Position: {E_Ts[0]}')
    print(f'Expected rise time for Position: {E_Tr[0]}')
    print(f'Expected overshoot estimate for Position: {E_OS[0]}')
    print(f'Expected crossing count for Position: {E_Osc[0]}')
    print(f'Expected settling time for Velocity: {E_Ts[1]}')
    print(f'Expected rise time for Velocity: {E_Tr[1]}')
    print(f'Expected overshoot estimate for Velocity: {E_OS[1]}')
    print(f'Expected crossing count for Velocity: {E_Osc[1]}')
    print(f'Expected peak value of control {E_Up[0]}')
    print(f'Expected integral of absolute value of control {E_Ui[0]}')

    plt.show()