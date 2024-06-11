# Function that takes in a state trajectory and returns a control trajectory according to the control law
import numpy as np
from simulate import simulate
import matplotlib.pyplot as plt 

def generate_control(control_law, solution_dict):
    """
        Calculates control values for each state in a simulation trajectory
        @ In, control_law, function, function of state x and time t
        @ In, solution_dict, dict, information from simulation run
        @ Out, u_t, np.array, control per time
    """
    # Retrieving information from solution
    tvec = solution_dict['t']
    states = solution_dict['y']

    # Initialize the output variable using control law
    u_0 = control_law(tvec[0], states[:,0])
    u_t = np.empty((len(u_0),len(tvec)))

    # Calculating the control for each state
    for index, state in enumerate(np.transpose(states)):
        u_t[:,index] = control_law(tvec[index], state)

    return u_t

def simple_open_loop(t, X, U):
    """
        Open loop undamped mass spring system with external control (m=k=1)
        @ In, t, float, current time
        @ In, X, np.array, current state
        @ In, U, np.array, current control
        @ Out, Xdot, np.array, dynamics
    """
    # Second order system rewritten as system of first order equations
    Xdot = np.empty(2)
    Xdot[0] = X[1]
    Xdot[1] = -1*X[0] + U[0]
    return Xdot

def feedback_control(t, X, K=np.array([1,1])):
    """
        Simple feed back control for simple example
        @ In, t, float, current time
        @ In, X, np.array, current state
        @ In, K, np.array, gain for controller
        @ Out, U, np.array, control value for current state
    """
    # Just from the dynamics lol
    U = np.empty(1)
    U[0] = (-1*K[0]*X[0]) - (K[1]*X[1])
    return U

if __name__ == '__main__':
    # Let's define the closed loop system to simulate the control laws
    # Uncontrolled
    control_1 = lambda t, X: np.zeros(1)
    closed_loop_1 = lambda t, y : simple_open_loop(t, X=y, U=control_1(t, X=y))
    # Controlled
    control_2 = lambda t, X: feedback_control(t, X)
    closed_loop_2 = lambda t, y : simple_open_loop(t, X=y, U=control_2(t, X=y))
    initial_conditions = np.array([1.5, -0.5])
    tsteps = np.linspace(0, 10, 1000)

    # Simulating controlled/uncontrolled systems
    solution_1 = simulate(closed_loop_1, initial_conditions, tsteps)
    solution_2 = simulate(closed_loop_2, initial_conditions, tsteps)

    # Using results to generate control outputs
    cont_1 = generate_control(control_1, solution_1)
    cont_2 = generate_control(control_2, solution_2)
    
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot the states at a function of time
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    # Plotting position
    ax1.plot(solution_1['t'], solution_1['y'][0][:], linewidth=1.5, color='blue')
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=14)
    ax1.grid()

    # Plotting velocity
    ax2.plot(solution_1['t'], solution_1['y'][1][:], linewidth=1.5, color='red')
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=14)
    ax2.grid()

    # Plotting control
    ax3.plot(solution_1['t'], cont_1[0,:], linewidth=1.5, color='green')
    ax3.set_xlabel('t', fontsize=14)
    ax3.set_ylabel('$u$ (Control)', fontsize=14)
    ax3.grid()

    fig.suptitle('Undamped Uncontrolled Oscillator System', fontsize=18)

    plt.show()

    # Let's plot the states at a function of time
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

    # Plotting position
    ax1.plot(solution_2['t'], solution_2['y'][0][:], linewidth=1.5, color='blue')
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=14)
    ax1.grid()

    # Plotting velocity
    ax2.plot(solution_2['t'], solution_2['y'][1][:], linewidth=1.5, color='red')
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=14)
    ax2.grid()

    # Plotting control
    ax3.plot(solution_2['t'], cont_2[0,:], linewidth=1.5, color='green')
    ax3.set_xlabel('t', fontsize=14)
    ax3.set_ylabel('$u$ (Control)', fontsize=14)
    ax3.grid()

    fig.suptitle('Undamped Controlled Oscillator System', fontsize=18)

    plt.show()