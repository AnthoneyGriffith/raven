# Script that provides the simulate system function, running as main tests functionality
import scipy.integrate as int
import numpy as np
import matplotlib.pyplot as plt

def simulate(closed_loop_dynamics, initial_conditions, tsteps, options=None):
    """
        Takes in closed-loop system and initial conditions, simulates, and returns the results
        @ In, closed_loop_dynamics, python function f(t,x), takes in time and state and returns dynamics
        @ In, initial_conditions, np.array, initial conditions for dynamics system in closed_loop_dynamics
        @ In, tsteps, np.array, set of times to return states over from integration
        @ In, options, dict, settings for RK45 integrator in scipy.integrate
        @ Out, solution, scipy.integrate solution object, results from integration
    """
    # Checking that dynamic system and intial condition make sense
    try:
        closed_loop_dynamics(tsteps[0], initial_conditions)
    except:
        print(f'There was an issue with the system dynamics and initial condition')

    # Integrate the system forward
    solution = int.solve_ivp(fun=closed_loop_dynamics, t_span=(tsteps[0], tsteps[-1]),
                          y0=initial_conditions, t_eval=tsteps, method='RK45')

    return solution

if __name__ == '__main__':
    print('Testing simulator for demonstration')

    # Using a harmonic oscillator to test functionality
    def harmonic_osc(t, x, u=0, omega_0=10, zeta=0.2, m=1):
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
    tsteps = np.linspace(0, 10, 1000)

    solution = simulate(closed_loop_dynamics, initial_conditions, tsteps)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot the states at a function of time
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # Plotting position
    ax1.plot(solution['t'], solution['y'][0][:], linewidth=1.5, color='blue')
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=14)
    ax1.grid()

    # Plotting velocity
    ax2.plot(solution['t'], solution['y'][1][:], linewidth=1.5, color='red')
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=14)
    ax2.grid()

    fig.suptitle('Damped Oscillator System', fontsize=18)

    plt.show()

    # Let's try and undamped system
    closed_loop_dynamics = lambda t, y: harmonic_osc(t, x=y, zeta=0)

    solution = simulate(closed_loop_dynamics, initial_conditions, tsteps)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot the states at a function of time
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # Plotting position
    ax1.plot(solution['t'], solution['y'][0][:], linewidth=1.5, color='blue')
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=14)
    ax1.grid()

    # Plotting velocity
    ax2.plot(solution['t'], solution['y'][1][:], linewidth=1.5, color='red')
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=14)
    ax2.grid()

    fig.suptitle('Undamped Oscillator System', fontsize=18)

    plt.show()

    # Why not a constant forcing function with a damped system, just for fun
    closed_loop_dynamics = lambda t, y: harmonic_osc(t, x=y, u=200000)

    solution = simulate(closed_loop_dynamics, initial_conditions, tsteps)

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    # Let's plot the states at a function of time
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # Plotting position
    ax1.plot(solution['t'], solution['y'][0][:], linewidth=1.5, color='blue')
    ax1.set_xlabel('t', fontsize=14)
    ax1.set_ylabel('$x_1$ (Position)', fontsize=14)
    ax1.grid()

    # Plotting velocity
    ax2.plot(solution['t'], solution['y'][1][:], linewidth=1.5, color='red')
    ax2.set_xlabel('t', fontsize=14)
    ax2.set_ylabel('$x_2$ (Velocity)', fontsize=14)
    ax2.grid()

    fig.suptitle('Forced and Damped Oscillator System', fontsize=18)

    plt.show()
