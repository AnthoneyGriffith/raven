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
    os.system(h_command)

    # Gotta find the outer file
    try:
        outer_slice = heron_input.rfind('/')
    except:
        outer_slice = heron_input.rfind('\\')
    outer_base = heron_input[0:outer_slice+1] + 'outer.xml'

    # Preprocess the outer file
    outer_new = preprocessOuter(outer_base, opt_params)

    # Looping over sample runs
    for samp in range(sample_count):
        # Just to see where we are at...
        print(f'Running trial {samp+1}...')

        # Update outer file for next trial
        trial_outer = updateOuter(outer_new, samp+1)

        # The raven command is then
        r_command = raven_loc + " " + trial_outer
        os.system(r_command)

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
    parallel = case.find('parallel')
    if opt_params['HPC']:
        if parallel is None:
            parallel = tree.SubElement(case, 'parallel')

        if opt_params['Inner Optimization Cores'] is not None:
            inner = parallel.find('inner')
            if inner is None:
                inner = tree.SubElement(parallel, 'inner')
            inner.text = opt_params['Inner Optimization Cores']
        out = parallel.find('outer')
        if out is None: 
            out = tree.SubElement(parallel, 'outer')
        out.text = str(1)

        runinfo = parallel.find('runinfo')
        if runinfo is None:
            runinfo = tree.SubElement(parallel,'runinfo')
            # Various parameters to edit
        time = runinfo.find('expectedTime')
        params = runinfo.find('clusterParameters')
        memory = runinfo.find('memory')
        if time is None:
            time = tree.SubElement(runinfo, 'expectedTime')
        if params is None:
            params = tree.SubElement(runinfo, 'clusterParameters')
        if memory is None:
            memory = tree.SubElement(runinfo, 'memory')
        # Updating node inputs
        time.text = opt_params['Max Runtime']
        params.text = '-P neup'
        memory.text = opt_params['Memory']
    else:
        if parallel is not None:
            inner = parallel.find('inner')
            if inner is not None:
                parallel.remove(inner)
            outer = parallel.find('outer')
            if outer is not None:
                parallel.remove(outer)
            runinfo = parallel.find('runinfo')
            if runinfo is not None:
                time = runinfo.find('expectedTime')
                if time is not None:
                    runinfo.remove(time)
                params = runinfo.find('clusterParameters')
                if params is not None:
                    runinfo.remove(params)
                memory = runinfo.find('memory')
                if memory is not None:
                    runinfo.remove(memory)
                parallel.remove(runinfo)
            case.remove(parallel)

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

    # Setting realization count and project life
    if opt_params['Realizations'] is not None:
        case.find('num_arma_samples').text = opt_params['Realizations']
    if opt_params['Project Life'] is not None:
        case.find('economics').find('ProjectTime').text = opt_params['Project Life']

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

    # Change base job name
    jobname = parsed.find('RunInfo').find('JobName')
    jobname.text = opt_params['Analysis Name']

    # Change working directory name
    workingdir = parsed.find('RunInfo').find('WorkingDir')
    workingdir.text = 'Opt_info' + '_' + opt_params['Analysis Name']

    # # Removing plot from sequence, steps, outstreams, etc
    # parsed.find('RunInfo').find('Sequence').text = 'optimize'
    # parsed.find('Steps').remove(parsed.find('Steps').find('IOStep'))
    # parsed.find('OutStreams').remove(parsed.find('OutStreams').find('Plot'))

    # Optimizer objects of BO and GD treated slightly different
    output = parsed.find('DataObjects').findall(".//PointSet/[@name='opt_soln']")[0].find('Output')
    opt = parsed.find("Optimizers")[0]
    if opt.tag != 'BayesianOptimizer':
        output.text = output.text + ', modelRuns, stepSize, rejectReason, conv_gradient, conv_samePoint, conv_objective'
        extension = '_' + opt_params['Analysis Name'] + '.xml'
        new_outer = outer_file.replace('.xml', extension)
        parsed.write(new_outer)
        return new_outer
    # Changing outputs to have everything
    output.text = output.text + ', solutionDeviation, rejectReason, modelRuns, radiusFromBest, radiusFromLast, solutionValue, acquisition'

    # Adding initials to variables for analysis
    variables = opt.findall("variable")
    for var in variables:
        initial = tree.SubElement(var,'initial')

    # Remove sampler object
    opt.remove(opt.find('Sampler'))

    # Editing seeding counts for suboptimization routines
    if opt_params['Model Seeds'] is not None:
        parsed.find('Models').findall(".//ROM/[@name='gpROM']")[0].find('n_restarts_optimizer').text = opt_params['Model Seeds']
    if opt_params['Acquisition Seeds'] is not None:
        opt.find('Acquisition')[0].find('seedingCount').text = opt_params['Acquisition Seeds']
    
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

    # Update job name
    jobname = parsed.find('RunInfo').find('JobName')
    jobname.text = jobname.text + '_' + str(current_trial)

    # New working directory so HPC doesn't yell at me anymore
    workingDir = parsed.find('RunInfo').find('WorkingDir')
    workingDir.text = workingDir.text + '_' + str(current_trial)

    # Distributions for variables
    dists = parsed.find("Distributions")
    var_dict = {}

    # # Updating solution export name in outstreams and steps
    # opt_out = parsed.find('OutStreams').findall(".//Print/[@name='opt_soln']")[0]
    # opt_out.attrib['name'] = opt_out.attrib['name'] + '_' + str(current_trial)
    # output_step = parsed.find('Steps').find('MultiRun').findall(".//Output/[@class='OutStreams']")[0]
    # output_step.text = opt_out.attrib['name']

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
   
    # Rename outer for current trial
    extension = '_' + str(current_trial) + '.xml'
    new_outer = outer_file.replace('.xml', extension)
    parsed.write(new_outer)
    return new_outer

if __name__ == '__main__':
    # Parsing input arguments to use
    parser = arg.ArgumentParser()
    # Arguments available
    parser.add_argument("-hpc", "--hpc", required=False, help='is this being ran through INL hpc? [y/n]')
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
    parser.add_argument("-ip", "--innerparallel", required=False, help='Number of cores for inner optimization')
    parser.add_argument("-re", "--realizations", required=False, help='Number of realizations for the TEA')
    parser.add_argument("-pl", "--life", required=False, help='Number of years for project life')
    parser.add_argument("-ms", "--modelseeds", required=False, help="Number of seedings for GPR model selection")
    parser.add_argument("-as", "--acquisitionseeds", required=False, help='Number of seeds for acquisition optimization')
    args = parser.parse_args()
    opt_params = {'Analysis Name':args.name,
                  'Max Evaluations':args.evals,
                  'Max Runtime':args.runtime,
                  'Memory':args.memory,
                  'Optimizer':args.optimizer,
                  'Kernel':args.kernel,
                  'Acquisition':args.acquisition,
                  'Inner Optimization Cores':args.innerparallel,
                  'Realizations':args.realizations,
                  'Project Life':args.life,
                  'Model Seeds':args.modelseeds,
                  'Acquisition Seeds':args.acquisitionseeds}
    if args.hpc is None or args.hpc == 'y':
        opt_params.update({'HPC':True})
    elif args.hpc == 'n':
        opt_params.update({'HPC':False})
    else:
        print('Invalid input for arg -hpc')
        exit()
    ravenLoop(args.raven, args.heron, args.input, int(args.trials), opt_params)
