# Python file for raven to evaluate system dynamics/optimize them
import sys
sys.path.append('..')
# Local files
from simulate import simulate
from evaluate_trajectory import evaluate_trajectory
from evaluate_multi import evaluate_multi
from multi_simulation import multi_simulation
from generate_control import generate_control
from open_loop import open_loop
from control_law import control_law
# Python libraries
import numpy as np
import matplotlib.pyplot as plt

def simulation(K):
    """
        Runs the simulation
    """
    # E_Ts_x1 = np.empty(len(K))
    # E_Ts_x2 = np.empty(len(K))
    # E_OS_x1 = np.empty(len(K))
    # E_OS_x2 = np.empty(len(K))
    # E_Osc_x1 = np.empty(len(K))
    # E_Osc_x2 = np.empty(len(K))
    tsteps = np.linspace(0, 20, 100)
    dist_bounds = np.array([[-5,5],[-1,1]])
    # try:
    #     test = len(K)
    #     for index, k in enumerate(K):
    #         control = lambda y: control_law(y, k)
    #         closed_loop_dynamics = lambda t, y: open_loop(t, y, control)

    #         multi, E_Ts, E_OS, E_Osc  = evaluate_multi(closed_loop_dynamics, tsteps, 'uniform',
    #                                 dist_bounds, sample_count=100, settling_time_tol=0.02
    #                                 )
    #         E_Ts_x1[index] = E_Ts[0]
    #         E_Ts_x2[index] = E_Ts[1]
    #         E_OS_x1[index] = E_OS[0]
    #         E_OS_x2[index] = E_OS[1]
    #         E_Osc_x1[index] = E_Osc[0]
    #         E_Osc_x2[index] = E_Osc[1]
    #     return E_Ts_x1, E_Ts_x2, E_OS_x1, E_OS_x2, E_Osc_x1, E_Osc_x2
    # except:
    control = lambda y: control_law(y, K)
    closed_loop_dynamics = lambda t, y: open_loop(t, y, control)

    multi, E_Ts, E_OS, E_Osc  = evaluate_multi(closed_loop_dynamics, tsteps, 'uniform',
                            dist_bounds, sample_count=100, settling_time_tol=0.02
                            )
    return E_Ts[0], E_Ts[1], E_OS[0], E_OS[1], E_Osc[0], E_Osc[1]


def run(self, Input):
    # Initializing
    # self.E_Ts_x1 = np.empty(len(self.k))
    # self.E_Ts_x2 = np.empty(len(self.k))
    # self.E_OS_x1 = np.empty(len(self.k))
    # self.E_OS_x2 = np.empty(len(self.k))
    # self.E_Osc_x1 = np.empty(len(self.k))
    # self.E_Osc_x2 = np.empty(len(self.k))
    # tsteps = np.linspace(0, 20, 100)
    # dist_bounds = np.array([[-5,5],[-1,1]])
    # for index, k in enumerate(self.k):
    #     control = lambda y: control_law(y, k)
    #     closed_loop_dynamics = lambda t, y: open_loop(t, y, control)

    #     multi, E_Ts, E_OS, E_Osc  = evaluate_multi(closed_loop_dynamics, tsteps, 'uniform',
    #                             dist_bounds, sample_count=100, settling_time_tol=0.02
    #                             )
    #     self.E_Ts_x1[index] = E_Ts[0]
    #     self.E_Ts_x2[index] = E_Ts[1]
    #     self.E_OS_x1[index] = E_OS[0]
    #     self.E_OS_x2[index] = E_OS[1]
    #     self.E_Osc_x1[index] = E_Osc[0]
    #     self.E_Osc_x2[index] = E_Osc[1]

    # Dummy testing
    # self.E_Ts_x1 = np.zeros(len(self.k))
    # self.E_Ts_x2 = np.zeros(len(self.k))
    # self.E_OS_x1 = np.zeros(len(self.k))
    # self.E_OS_x2 = np.zeros(len(self.k))
    # self.E_Osc_x1 = np.zeros(len(self.k))
    # self.E_Osc_x2 = np.zeros(len(self.k))

    self.E_Ts_x1, self.E_Ts_x2, self.E_OS_x1, self.E_OS_x2, self.E_Osc_x1, self.E_Osc_x2 = simulation(self.k)
