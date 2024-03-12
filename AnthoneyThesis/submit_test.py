# This file submits all jobs necessary for a given use case for BO comparison
import os
import sys
import subprocess as sub
import argparse as arg
import time

def submit(arg_dict, opt_dict, acqu_dict):
    """
        Loops over combinations of optimizers for a given use case
        @ In, arg_dict, inputs to build tests from
        @ In, opt_dict, optimizer kernels with names
        @ In, acqu_dict, acquisition functions with names
    """
    # Let's collect the things that will always be in the command
    base_command = ['python', 'testing_loop_hpc.py',
                    '-r', arg_dict['Raven'],
                    '-he', arg_dict['Heron'],
                    '-i', arg_dict['Input'],
                    '-t', arg_dict['Trials'],
                    '-e', arg_dict['Evaluations'],
                    '-pl', arg_dict['Project Life'],
                    '-re', arg_dict['Realizations'],
                    '-o', 'BayesianOptimizer',
                    '-m', '10g',
                    '-rt', '140:00:00',
                    '-as', str(int(25*float(arg_dict['Dimension']))),
                    '-ip', str(1),
                    '-d', str(1),
                    ]
    # Time to iterate through optimizer variations
    for k_name, kernel in opt_dict.items():
        for a_name, acqu in acqu_dict.items():
            new_command = base_command.copy()
            name = arg_dict['Case Name'] + '_' + k_name + '_' + a_name
            modelseeds = str(5*kernel[1])
            kern = kernel[0]
            acquisition = acqu
            new_command.append('-n')
            new_command.append(name)
            new_command.append('-ms')
            new_command.append(modelseeds)
            new_command.append('-k')
            new_command.append(kern)
            new_command.append('-a')
            new_command.append(acquisition)
            print(f'Submitting jobs for {name}...\n')
            print(' '.join(new_command))
            time.sleep(5)
            call = sub.run(new_command, stdout=sub.PIPE, text=False)

if __name__ == '__main__':
    # Parsing input arguments to use
    parser = arg.ArgumentParser()

    # Arguments available
    parser.add_argument("-r", "--raven", required=True, help='raven_framework file for running RAVEN')
    parser.add_argument("-he", "--heron", required=True, help='heron file for running HERON')
    parser.add_argument("-i", "--input", required=True, help='heron input file for TEA')
    parser.add_argument("-t", "--trials", required=True, help='number of trials')
    parser.add_argument("-e", "--evals", required=True, help='numer of model evaluations per optimizatin')
    parser.add_argument("-n", "--name", required=True, help='Name of use case')
    parser.add_argument("-re", "--realizations", required=False, help='Number of realizations for the TEA')
    parser.add_argument("-pl", "--life", required=False, help='Number of years for project life')
    parser.add_argument("-d", "--dimension", required=False, help='Number of capacities to select')
    args = parser.parse_args()

    arg_dict = {'Raven':args.raven,
                'Heron':args.heron,
                'Input':args.input,
                'Trials':args.trials,
                'Evaluations':args.evals,
                'Case Name':args.name,
                'Realizations':args.realizations,
                'Project Life':args.life,
                'Dimension':int(args.dimension)}
    
    opt_dict = {'RBF':['Constant*RBF', float(args.dimension)+1],
                'M':['Constant*Matern', float(args.dimension)+1],
                'RQ':['Constant*RationalQuadratic', float(args.dimension)+1],
                'MW':['Constant*Matern+WhiteNoise', float(args.dimension)+2]}
    
    acqu_dict = {'EI':'ExpectedImprovement',
                 'PoI':'ProbabilityOfImprovement',
                 'LCB':'LowerConfidenceBound'}
    
    submit(arg_dict, opt_dict, acqu_dict)
