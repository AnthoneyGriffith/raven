# import os
# import sys
# sys.path.append('raven')
# sys.path.append('HERON/tests/integration_tests/mechanics/')

# # instantiate a RAVEN instance
# from ravenframework import Raven
# raven = Raven()

# # Load optimization workflow
# raven.loadWorkflowFromFile('outer.xml')

# # run the workflow
# returnCode = raven.runWorkflow()
# # check for successful run
# if returnCode != 0:
#   raise RuntimeError('RAVEN did not run successfully!')

import os
import sys
import pandas as pd
import xml.etree.ElementTree as tree
import numpy as np
import platform as plat
import subprocess as sub

def ravenLoop(raven, outer_dir, csv_dir, sample_count, solution_dir, heron_dir):
    """
        Runs Raven in a loop for the sake of generating and storing runs for cumulative comparison.
        Changes initial optimization points between runs and is meant to use 'thesis' branch of
        Raven.
        @ In, raven, str, absolute directory of raven
        @ In, outer_dir, str, absolute directory to outer.xml file for running
        @ In, csv_dir, str, location to save csv files to
        @ In, sample_count, number of samples to run analysis for
        @ In, solution_dir, name of csv where solutions are stored
        @ In, heron_dir, name of heron input to run to set synthetic history model
    """
    # Adding raven running script to raven directory, correcting paths
    os_home = os.path.expanduser("~")
    raven = raven.replace("~", os_home)
    outer_dir = outer_dir.replace("~", os_home)
    solution_dir = solution_dir.replace("~", os_home)
    heron_dir = heron_dir.replace("~", os_home)

    # Assumes a relative location for HERON
    heron_command = raven + "..\HERON\heron " + heron_dir
    raven_command = raven + "raven_framework " + outer_dir

    # Running HERON to update the inner details with correct directory for armas
    os.system(heron_command)

    # Need to make sure correct raven executable is used in outer
    addExecutableToOuter(raven, outer_dir)

    # Looping over sample runs
    for samp in range(sample_count):
        print(f'Running trial {samp+1}...')
        # If to avoid running on crappy windows for now
        if plat.system() != 'Windows':
            os.system(raven_command)
        else:
            print('Under construction, likely will not finish unless I have to')
            exit()
            git_code =  "C:\Program Files\Git\git-bash.exe"
            windows_cmd = git_code + " " + raven_command
            p = sub.Popen(windows_cmd, 
                 bufsize=-1, 
                 executable=None, 
                 stdin=None, 
                 stdout=None, 
                 stderr=None, 
                 preexec_fn=None, 
                 close_fds=False, 
                 shell=False, 
                 cwd=outer_dir, 
                 )
            p.wait()

        # Saving solution as next csv to store
        dataframe = pd.read_csv(solution_dir)
        csv_loc = csv_dir + f'\Opt_{samp+1}.csv'
        dataframe.to_csv(csv_loc)
        
        updateOuterInitialPoints(outer_dir)

def addExecutableToOuter(raven, outer_file):
    """
        Parses and updates outer file necessary directories
        @ In, raven, str, absolute directory of raven
        @ In, outer_file, outer.xml to edit the initial points in optimization for
    """
    # Parsing outer
    parsed = tree.parse(outer_file)
    # Models in outer
    models = parsed.find("Models")
    # RAVEN code node is where we want to change executable
    code = models.findall(".//executable/..[@name='raven']")
    if len(code) != 1:
        print('Something is very wrong with outer, please fix: No Raven code model found, or multiple defined.')
        exit()
    raven_code = code[0]
    # Changing executable to correct thing
    raven_code.find('executable').text = raven + 'raven_framework'
    # Updating outer file
    parsed.write(outer_file)

def updateOuterInitialPoints(outer_file):
    """
        Parses and updates outer file optimizer initial points
        @ In, outer_file, outer.xml to edit the initial points in optimization for
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
    parsed.write(outer_file)

if __name__ == '__main__':
    raven = sys.argv[1]
    outer_dir = sys.argv[2]
    csv_dir = sys.argv[3]
    sample_count = int(sys.argv[4])
    solution_dir = sys.argv[5]
    heron_dir = sys.argv[6]
    ravenLoop(raven, outer_dir, csv_dir, sample_count, solution_dir, heron_dir)
