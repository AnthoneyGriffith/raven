# Python file for raven to evaluate system dynamics/optimize them
import sys
sys.path.append('..')
# Main function call tool for analysis
from evaluate_multi import evaluate_multi
from generate_control import simple_open_loop, feedback_control
# Python libraries (could be useful)
import numpy as np
import matplotlib.pyplot as plt

def build_closed_loop_dynamics(Input):
  """
    Takes in info from RAVEN and sets controller accordingly
    @ In, Input, input from RAVEN, this should be the Gains in particular
    @ Out, closed_loop, lambda function, takes in t and y and outputs ydot for closed loop system
    @ Out, control_law, lambda function, takes in t and y and outputs u
  """
  # Retrieve gains
  k1 = float(Input['k1'])
  k2 = float(Input['k2'])

  # Specify controller
  control_law = lambda t, y: feedback_control(t, X=y, K=np.array([k1, k2]))

  # Specify closed_loop_dynamics
  closed_loop = lambda t, y: simple_open_loop(t, X=y, U=control_law(t, y))

  return closed_loop, control_law

def simulation(self, Input, closed_loop, control):
  """
    Main method that RAVEN calls when sampling
    @ In, self, raven evaluation object
    @ In, Input, set of gains from RAVEN
    @ In, closed_loop, resulting closed loop dynamics
    @ In, control, control function
    @ Out, None
  """
  # Running monte carlo simulation of closed-loop system
  tsteps = np.linspace(0, 20, 20000)
  initial_condition_dist = 'uniform'
  dist_bounds = np.array([[-2, 2], [-3, 3]])
  sample_count = 200
  settling_time_tol = 0.05
  rise_time_ratio = 0.05
  desired_state = None
  multi_traj, multi_ctrl, E_Ts, E_Tr, E_OS, E_Osc, E_Up, E_Ui = evaluate_multi(closed_loop, tsteps, initial_condition_dist, dist_bounds, sample_count, settling_time_tol,
                                                                               rise_time_ratio, desired_state, control_law=control)
  self.E_Ts_x1 = E_Ts[0]
  self.E_Tr_x1 = E_Tr[0]
  self.E_OS_x1 = E_OS[0]
  self.E_Osc_x1 = E_Osc[0]
  self.E_Ts_x2 = E_Ts[1]
  self.E_Tr_x2 = E_Tr[1]
  self.E_OS_x2 = E_OS[1]
  self.E_Osc_x2 = E_Osc[1]
  self.E_Up = E_Up[0]
  self.E_Ui = E_Ui[0]
  W_Ts = 1/5
  W_Tr = 1/3
  W_OS = 1/4
  W_Osc = 1/13
  W_Up = 1
  W_Ui = 1/10
  term1 = np.add(np.multiply(W_Ts, np.sum(E_Ts)), np.multiply(W_Tr, np.sum(E_Tr)))
  term2 = np.add(np.multiply(W_OS, np.sum(E_OS)), np.multiply(W_Osc, np.sum(E_Osc)))
  term3 = np.add(np.multiply(W_Up, np.sum(E_Up)), np.multiply(W_Ui, np.sum(E_Ui)))
  self.J = term1 + term2 + term3

def run(self, Input):
  """
    Main method that RAVEN calls when sampling
    @ In, self, raven evaluation object
    @ In, Input, set of gains from RAVEN
    @ Out, None
  """
  # Generate closed loop dynamics according to input from RAVEN
  closed_loop, control_law = build_closed_loop_dynamics(Input)
  simulation(self, Input, closed_loop, control_law)
