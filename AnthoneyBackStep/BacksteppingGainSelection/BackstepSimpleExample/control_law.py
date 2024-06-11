# Function for control law
import sys
sys.path.append('..')
from simulate import simulate
from generate_control import generate_control
from open_loop import open_loop
import numpy as np
import matplotlib.pyplot as plt

def control_law(x, k):
    """
        @ In, x, state
        @ In, k, gain
        @ Out, u, control
    """
    # z state from backstepping approach
    z = x[1] + x[0] + x[0]**2
    u = np.array([-1*x[0] - (1 + 2*x[0])*(-1*x[0] - x[0]**3 + z) - k*z])
    return u

if __name__ == '__main__':
    # Simulating the dynamics here for sanity's sake
    k = 3.5e1
    control_gen = lambda t, y: control_law(y, k)
    control = lambda y: control_law(y, k)
    dynamics = lambda t, y: open_loop(t, y, control)
    initial_conditions = np.array([5, 0])
    tsteps = np.linspace(0, 10, 1000)
    sol = simulate(dynamics, initial_conditions, tsteps)
    ctrl = generate_control(control_gen, sol)

    # Let's plot
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(sol['t'], sol['y'][0,:], color='blue', linewidth=1.5)
    ax2.plot(sol['t'], sol['y'][1,:], color='red', linewidth=1.5)
    ax3.plot(sol['t'], ctrl[0,:], color='green', linewidth=1.5)
    ax1.set_ylabel('$x_1$', fontsize=16)
    ax2.set_ylabel('$x_2$', fontsize=16)
    ax3.set_ylabel('$u$', fontsize=16)
    ax3.set_xlabel('$t$', fontsize=16)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()
