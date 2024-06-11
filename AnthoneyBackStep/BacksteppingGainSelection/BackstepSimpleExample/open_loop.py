# Open loop dynamics for the simple example Harris provided in class
import sys
sys.path.append('..')
from simulate import simulate
import numpy as np
import matplotlib.pyplot as plt

def open_loop(t, x, control_law):
    """
        Open loop dynamics of simple example
        @ In, t, time
        @ In, x, state
        @ In, control_law, function of the state
        @ Out, xdot, dynamics
    """
    # Just some simple dynamics
    xdot = np.empty(len(x))
    xdot[0] = x[0]**2 - x[0]**3 + x[1]
    xdot[1] = control_law(x)
    return xdot

if __name__ == '__main__':
    # Simulating the dynamics here for sanity's sake
    control = lambda y: 0
    dynamics = lambda t, y: open_loop(t, y, control)
    initial_conditions = np.array([5, 0])
    tsteps = np.linspace(0, 10, 1000)
    sol = simulate(dynamics, initial_conditions, tsteps)

    # Let's plot
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(sol['t'], sol['y'][0,:], color='blue', linewidth=1.5)
    ax2.plot(sol['t'], sol['y'][1,:], color='red', linewidth=1.5)
    ax1.set_ylabel('$x_1$', fontsize=16)
    ax2.set_ylabel('$x_2$', fontsize=16)
    ax2.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()
    plt.show()