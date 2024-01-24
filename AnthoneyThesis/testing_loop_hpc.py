# Python file for sequentially submitting jobs to HPC for testing BO and GD on the TEA
import os
import sys
import pandas as pd
import xml.etree.ElementTree as tree
import numpy as np
import platform as plat
import subprocess as sub
import argparse as arg

def ravenLoop(raven_loc, heron_loc, heron_input, sample_count, opt_params):
    """
        Runs Raven in a loop for the sake of generating and storing runs for cumulative comparison.
        Changes initial optimization points between runs and is meant to use 'thesis' branch of
        Raven.
        @ In, raven_loc, absolute path to raven_framework script for running raven
        @ In, heron_loc, absolute path to heron script for running heron
        @ In, heron_input, absolute path to heron input file for analysis
        @ In, sample_count, int, number of trials to submit to HPC
        @ In, opt_params, dict, additional information for editing xmls
    """
    # Ensuring paths to various locations are correct for current computer
    os_home = os.path.expanduser("~")
    raven_loc = raven_loc.replace("~", os_home)
    heron_loc = heron_loc.replace("~", os_home)
    heron_input = heron_input.replace("~", os_home)

    # Need make new heron input with correct qsub parameters and arma directories
    new_heron = rewriteHeronInput(heron_input, opt_params)
    h_command = heron_loc + " " + new_heron
    os.sys(h_command)

    # Gotta find the outer file
    try:
        outer_slice = heron_input.rfind('/')
    except:
        outer_slice = heron_input.rfind('\\')
    outer_base = heron_input[0:outer_slice+1] + 'outer.xml'
    
    # Preprocess outer if it is BO, otherwise we gucci
    heron_parsed = tree.parse(new_heron)
    strat = heron_parsed.find('Case').find('strategy')
    if strat is not None:
        if strat.text == 'BayesianOptimizer':
            outer_new = preprocessOuter(outer_base, opt_params)
        else:
            outer_new = outer_base
    else:
        outer_new = outer_base

    # Looping over sample runs
    for samp in range(sample_count):
        # Just to see where we are at...
        print(f'Running trial {samp+1}...')

        # Update outer file for next trial
        trial_outer = updateOuter(outer_new)

        # The raven command is then
        r_command = raven_loc + " " + trial_outer
        os.sys(r_command)

def rewriteHeronInput(heron_input, opt_params):
    """
        Updates information on heron input for qsub runs on hpc
        @ In, heron_input, str, directory of base heron_input
        @ In, opt_params, dict, additional information for editing xmls
        @ Out, new_input, str, location of new heron input
    """
    # Open current heron xml
    parsed = tree.parse(heron_input)
    # Looking for the case node
    case = parsed.find('Case')
    try:
        # Accessing the runinfo node
        runinfo = case.find('runinfo')
        # Various parameters to edit
        time = runinfo.find('expectedTime')
        params = runinfo.find('clusterParameters')
        memory = runinfo.find('memory')
    except:
        runinfo = tree.SubElement(case,'runinfo')
        time = tree.SubElement(runinfo, 'expectedTime')
        params = tree.SubElement(runinfo, 'clusterParameters')
        memory = tree.SubElement(runinfo, 'memory')
    # Updating node inputs
    time.text = opt_params['Max Runtime']
    params.text = '-P neup'
    memory.text = opt_params['Memory']
    # Updating optimization settings
    opt_settings = case.find('optimization_settings')
    # Number limit for optimizer, setting persistence higher to avoid premature convergence
    try:
        opt_settings.find('limit').text = opt_params['Max Evaluations']
    except:
        limit = tree.SubElement(opt_settings, 'limit')
        limit.text = opt_params['Max Evaluations']
    opt_settings.find('persistence').text = str(int(opt_params['Max Evaluations'])+1)
    # If optimizer is provided in args, set strategy
    if opt_params['Optimizer'] is not None:
        try:
            case.find('strategy').text = opt_params['Optimizer']
        except:
            strat = tree.SubElement(case, 'strategy')
            strat.text = opt_params['Optimizer']
    # Checking if BO
    strat = case.find('strategy')
    if strat is not None:
        if strat.text == 'BayesianOptimizer':
            if opt_params['Kernel'] is not None:
                try:
                    opt_settings.find('kernel').text = opt_params['Kernel']
                except:
                    kernel = tree.SubElement(opt_settings, 'kernel')
                    kernel.text = opt_params['Kernel']
            if opt_params['Acquisition'] is not None:
                try:
                    opt_settings.find('acquisition').text = opt_params['Acquisition']
                except:
                    acqu = tree.SubElement(opt_settings, 'acquisition')
                    acqu.text = opt_params['Acquisition']
        else:
            print('Gradient Descent is being used, no ability to set options available yet')

    # Since outers will have working dir one layer deeper for organization
    arma = parsed.find('DataGenerators').find('ARMA')
    if '%BASE_WORKING_DIR%' in arma.text:
        arma.text = arma.text.replace('%BASE_WORKING_DIR%', '%BASE_WORKING_DIR%/..')

    # Saving as a new heron input for just this trial
    input_extension = '_' + opt_params['Analysis Name'] + '.xml'
    new_input = heron_input.replace('.xml', input_extension)
    parsed.write(new_input)
    return new_input

def preprocessOuter(outer_file, opt_params):
    """
        Updates BO outers to have initial points and no sampler
        @ In, outer_file, str, directory of base outer.xml
        @ In, opt_params, dict, additional information for editing xmls
        @ Out, new_outer, str, location of new outer.xml
    """
    # Step one is to parse
    parsed = tree.parse(outer_file)
    # Optimizer object
    opt = parsed.find("Optimizers")[0]
    if opt.tag != 'BayesianOptimizer':
        print('There has been a failure in generating the correct outer.')
        exit()
    variables = opt.findall("variable")
    for var in variables:
        initial = tree.SubElement(var,'initial')
    # Remove sampler object
    opt.remove(opt.find('Sampler'))

    # Resave outer as unique thing
    extension = '_' + opt_params['Analysis Name'] + '.xml'
    new_outer = outer_file.replace('.xml', extension)
    parsed.write(new_outer)
    return new_outer

def updateOuter(outer_file, current_trial):
    """
        Parses and updates outer file optimizer initial points
        @ In, outer_file, outer.xml to edit the initial points in optimization for
        @ In, current_trial, int, current sample number
        @ Out, new_outer, str, name of edited outer
    """
    # Parse the xml file
    parsed = tree.parse(outer_file)
    # Distributions for variables
    dists = parsed.find("Distributions")
    var_dict = {}
    # Retrieving variable information
    for dist in dists:
        name = dist.attrib['name']
        bounds = np.empty(2)
        for bound in dist:
            if bound.tag == 'lowerBound':
                index = 0
            elif bound.tag == 'upperBound':
                index = 1
            else:
                continue
            bounds[index] = float(bound.text)
        var_dict.update({name:bounds})
    # Optimizer objects
    opt = parsed.find("Optimizers")[0]
    variables = opt.findall("variable")
    for var in variables:
        nametag = var.attrib['name'] + '_dist'
        sampling_bound = var_dict[nametag]
        var.find('initial').text = str(np.random.uniform(low=sampling_bound[0], high=sampling_bound[1]))
    # Update working directory
    working_dir = parsed.find('RunInfo').find('WorkingDir')
    working_dir.text = working_dir.text + '/OptRun' + str(current_trial)
    extension = '_' + str(current_trial) + '.xml'
    new_outer = outer_file.replace('.xml', extension)
    parsed.write(new_outer)
    return new_outer

if __name__ == '__main__':
    # Parsing input arguments to use
    parser = arg.ArgumentParser()
    # Arguments available
    parser.add_argument("-r", "--raven", required=True, help='raven_framework file for running RAVEN')
    parser.add_argument("-he", "--heron", required=True, help='heron file for running HERON')
    parser.add_argument("-i", "--input", required=True, help='heron input file for TEA')
    parser.add_argument("-t", "--trials", required=True, help='number of trials')
    parser.add_argument("-e", "--evals", required=True, help='numer of model evaluations per optimizatin')
    parser.add_argument("-rt", "--runtime", required=True, help='walltime allowed for each outer job')
    parser.add_argument("-m", "--memory", required=True, help='memory per hpc core')
    parser.add_argument("-k", "--kernel", required=False, help='kernel for BO')
    parser.add_argument("-a", "--acquisition", required=False, help='acquisition function for BO')
    parser.add_argument("-o", "--optimizer", required=False, help='BayesianOptimizer or GradientDescent')
    parser.add_argument("-n", "--name", required=True, help='Name of analysis')
    args = parser.parse_args()
    opt_params = {'Analysis Name':args.name,
                  'Max Evaluations':args.evals,
                  'Max Runtime':args.runtime,
                  'Memory':args.memory,
                  'Optimizer':args.optimizer,
                  'Kernel':args.kernel,
                  'Acquisition':args.acquisition}
    ravenLoop(args.raven, args.heron, args.input, int(args.trials), opt_params)
