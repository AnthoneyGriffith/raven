# Copyright 2023 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
"""
# External Modules
import scipy.optimize as sciopt
# External Modules

# Internal Modules
import abc
from ...utils import utils, InputData, InputTypes
# Internal Modules

class AcquisitionFunction(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Provides Base class for acquisition functions. Holds general methods for
    optimization of the acquisition functions
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class for acquisition functions for Bayesian Optimizer.'
    specs.addSub(InputData.parameterInputFactory('optimizationMethod', contentType=InputTypes.StringType,
        descr=r"""String to specify routine used for the optimization of the acquisition function.
              Acceptable options include ('differentialEvolution', 'slsqp'). \default{'differentialEvolution'}"""))
    specs.addSub(InputData.parameterInputFactory('seedingCount', contentType=InputTypes.IntegerType,
        descr=r"""This can describes two different but similar things depending on the selection of
              the optimization method. If the method is gradient based or typically handled with singular
              decisions (ex. slsqp approximates a quadratic program using ), this number
              represents the number of trajectories for a multi-start variant (default=2N).
              N is the dimension of the input space.
              If the method works on populations (ex. differential evolution), the number
              represents the population size (default=10N)."""))
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, vars, dict, acceptable variable names and descriptions
    """
    return {}

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self._optMethod = None  # Method used to optimize acquisition function for sample selection
    self._seedingCount = 0  # For multi-start gradient methods, the number of starting points and the population size for differential evolution
    self.N = None           # Dimension of the input space
    self._bounds = []       # List of tuples for bounds that scipy optimizers use
    self._optValue = None   # Value of the acquisition function at the recommended sample

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    settings, notFound = specs.findNodesAndExtractValues(['optimizationMethod', 'seedingCount'])
    # If no user provided setting for opt method and seeding count, use default
    if 'optimizationMethod' in notFound:
      self._optMethod = 'differential evolution'
    else:
      self._optMethod = settings['optimizationMethod']
    if 'seedingCount' in notFound:
      if self._optMethod == 'differential evolution':
        self._seedingCount = 10*self.N
      else:
        self._seedingCount = 2*self.N
    else:
      self._seedingCount = settings['seedingCount']

  def initialize(self):
    """
      After construction, finishes initialization of this acquisition function.
      @ In, None
      @ Out, None
    """
    # Input space is normalized, thus building the bounds is simple
    for i in range(self.N):
      self._bounds.append((0,1))
    # TODO should we feed explicit constraints here?

  def conductAcquisition(self, bayesianOptimizer):
    """
      Selects new sample via optimizing the acquisition function
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, newPoint, dict, new point to sample the cost function at
    """
    # Depending on the optimization method, the cost function should be defined differently
    if self._optMethod == 'differential evolution':
      # NOTE -1 is to enforce maximization of the positive function
      opt_func = lambda var: -1*self.evaluate(var, bayesianOptimizer, vectorized=True)
      res = sciopt.differential_evolution(opt_func, bounds=self._bounds, polish=True, maxiter=100, tol=1e-5,
                                          popsize=self._seedingCount, init='latinhypercube', vectorized=True)
    else:
      self.raiseAnError(RuntimeError, 'Currently only accepts differential evolution. Other methods still under construction')
    self._optValue = -1*res.fun
    newPoint = bayesianOptimizer.arrayToFeaturePoint(res.x)
    return newPoint

  ######################
  # Evaluation Methods #
  ######################
  @abc.abstractmethod
  def evaluate(self, var, bayesianOptimizer, vectorized=False):
    """
      Evaluates acquisition function using the current BO instance
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ In, vectorized, bool, whether the evaluation should be vectorized or not (useful for differential evolution)
      @ Out, acqValue, float/array, acquisition function value
    """

  @abc.abstractmethod
  def gradient(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's gradient using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function gradient at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, dacqValue, float/array, acquisition function gradient value
    """

  @abc.abstractmethod
  def hessian(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's hessian using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function hessian at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, ddacqValue, float/array, acquisition function hessian value
    """

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return False

  def updateSolutionExport(self):
    """
      Prints information to the solution export.
      @ In, None
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    # Returning acquisition value post optimization
    info = {'acquisition':self._optValue}
    self._optValue = None # Resetting
    return info

  ###################
  # Utility Methods #
  ###################
  def flush(self):
    """
      Reset GradientApproximater attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    self._optMethod = None  # Method used to optimize acquisition function for sample selection
    self._seedingCount = 0  # For multi-start gradient methods, the number of starting points and the population size for differential evolution
    self.N = None           # Dimension of the input space
    self._bounds = []       # List of tuples for bounds that scipy optimizers use
    self._optValue = None   # Value of the acquisition function at the recommended sample
    return
