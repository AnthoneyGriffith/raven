import pandas as pd
import os 
import sys
import argparse as arg

def retrieveDataCSVs(test_name, analysis_dir, goal_dir, trial_count, max_eval):
    """
        Skims through output directories to store csvs in a more convenient location
        @ In, test_name, str, name of analysis ran for thesis work
        @ In, analysis_dir, str, directory of analysis file
        @ In, goal_dir, str, where I want to save csvs
        @ In, trial_count, int, number of trials ran for this analysis
        @ In, max_eval, str, number of model evaluations expected by successful job
    """
    # Number of actual successful runs
    true_count = 0
    # Looping over trials and moving the csvs
    for samp in range(trial_count):
        # Just wanna track which ones I am loading
        print(f'Attempting to load and resave results for trial {samp+1}')
        # Abusing fixed naming structure with test name
        solution_dir = analysis_dir + '/' + test_name + '_' + str(samp+1) + "/Opt_info_" + test_name + '_' + str(samp+1)
        # Need to check if this job actually finished
        out_inner = solution_dir + '/optimize/' + max_eval + '/out~inner'

        # Checking if out inner exists for final evaluation
        finished = os.path.exists(out_inner)
        if finished:
            # Notify user
            print(f'Trial {samp+1} for test {test_name} appears to have finished...')
            print('Collecting csv...\n')
            solution_csv = solution_dir + '/opt_soln_0.csv'

            # Name of saved file in location I want
            to_save = goal_dir + '/Opt_' + str(true_count+1) + '.csv'
            # Loading and then saving as
            loaded = pd.read_csv(solution_csv)
            loaded.to_csv(to_save)
            true_count += 1
            print('Its working??')
            exit()
        else:
            print(f'Could not find out~inner for trial {samp+1}, assume it has failed.\n')
    print(f'All finished!\n'
          f'Out of a possible {trial_count} jobs,\n'
          f'{true_count} have finished and had their results retrieved')

if __name__ == '__main__':
    # Parsing input arguments
    parser = arg.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help='name of test to move data for')
    parser.add_argument("-l", "--location", required=True, help='directory of heron files where working directories are')
    parser.add_argument("-g", "--goal", required=True, help='desired directory for storing output files')
    parser.add_argument("-t", "--trials", required=True, help='number of trials for which folders are available')
    parser.add_argument("-e", "--evals", required=True, help='Number of expected evaluations for each trial')
    args = parser.parse_args()
    retrieveDataCSVs(args.name, args.location, args.goal, int(args.trials), args.evals)
