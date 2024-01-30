# File for clearing outers, heron inputs, and working dirs of a given test 
import os 
import sys
import argparse as arg

def deleteExcess(test_name, input_loc, trial_count):
    """
        Skims through directory and deletes the files created for trial runs.
        Just helps keep my HPC stuff clean so I don't get yelled at by INL
        @ In, test_name, name of test used in loop python file
        @ In, input_loc, location of heron input, used for finding files to delete
        @ In, trial_count, number of trials for this test
    """
    print('Under Construction')

if __name__ == '__main__':
    # Parsing input arguments
    parser = arg.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help='name of test to move data for')
    parser.add_argument("-l", "--location", required=True, help='directory of heron files where working directories are')
    parser.add_argument("-t", "--trials", required=True, help='number of trials for which folders are available')
    args = parser.parse_args()
    deleteExcess(args.name, args.location, int(args.trials))
    