# Calculates performance metrics for a given control like peak and approximate integral
import numpy as np
from simulate import simulate
from generate_control import generate_control, simple_open_loop, feedback_control
import matplotlib.pyplot as plt 

def evaluate_control(control_law, trajectory_info):
    """
        Generates control for simulation and calculates metrics of interest
        @ In, control_law, function, function of state x and time t
        @ In, trajectory_info, ode solution dict, information from simulation
        @ Out, Up, np.array, peak control used for each control input
        @ Out, Ui, np.array, approximate integral of |u| for each control input
        @ Out, control_vec, np.array, control trajectory from simulation results
    """
    # Need to determine the control
    control_vec = generate_control(control_law, solution_dict=trajectory_info)
    tvec = trajectory_info['t']

    # Initialize outputs
    Up = np.empty(len(control_vec[:,0]))
    Ui = np.empty(len(control_vec[:,0]))

    # Iterate through control and calculate metrics
    ctrl_index = 0
    for control in control_vec:
        # Calculating peak control value
        Up[ctrl_index] = float(np.max(np.abs(control)))

        # Calculating approximate integral of control w/ left sided Riemann sum
        Ui[ctrl_index] = 0
        reduced = len(control)-1
        for t_ind, ctrl in enumerate(control[0:reduced]):
            dt = tvec[t_ind+1] - tvec[t_ind]
            Ui[ctrl_index] += float(np.abs(np.multiply(dt,ctrl)))
        
        # Onto the next control variable
        ctrl_index += 1
    
    return Up, Ui, control_vec

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

    # Evaluating control
    Up1, Ui1 = evaluate_control(control_1, solution_1)
    print(f'The peak control value is {Up1[0]}')
    print(f'The integral of the absolute value of control is {Ui1[0]}')
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

    Up2, Ui2 = evaluate_control(control_2, solution_2)
    print(f'The peak control value is {Up2[0]}')
    print(f'The integral of the absolute value of control is {Ui2[0]}')
    plt.show()