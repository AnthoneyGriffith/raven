import pandas as pd
import os 
import sys
import argparse as arg

def retrieveDataCSVs(test_name, analysis_dir, goal_dir, trial_count):
    """
        Skims through output directories to store csvs in a more convenient location
        @ In, test_name, str, name of analysis ran for thesis work
        @ In, analysis_dir, str, directory of analysis file
        @ In, goal_dir, str, where I want to save csvs
        @ In, trial_count, int, number of trials ran for this analysis
    """
    # Looping over trials and moving the csvs
    for samp in range(trial_count):
        # Just wanna track which ones I am loading
        print(f'Loading and resaving results for trial {samp+1}')
        # Abusing fixed naming structure with test name
        solution_dir = analysis_dir + "Opt_info_" + test_name + '_' + str(samp+1)
        solution_csv = solution_dir + '/opt_soln_0.csv'

        # Name of saved file in location I want
        to_save = goal_dir + '/Opt_' + str(samp+1) + '.csv'

        # Loading and then saving as
        loaded = pd.read_csv(solution_csv)
        loaded.to_csv(to_save)
    print('All finished!')

if __name__ == '__main__':
    # Parsing input arguments
    parser = arg.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help='name of test to move data for')
    parser.add_argument("-l", "--location", required=True, help='directory of heron files where working directories are')
    parser.add_argument("-g", "--goal", required=True, help='desired directory for storing output files')
    parser.add_argument("-t", "--trials", required=True, help='number of trials for which folders are available')
    args = parser.parse_args()
    retrieveDataCSVs(args.name, args.location, args.goal, int(args.trials))
