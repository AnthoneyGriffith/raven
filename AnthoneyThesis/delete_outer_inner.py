# Deletes out inners that are eating up so much space
import os 
import sys
import argparse as arg

def clear_outer_inners(name, location, trials):
    """
        Deletes as many outer-inners as possible to save that precious storage space
        @ In, name of the test
        @ In, location of the test directory
        @ In, number of trials for the test
    """
    # Iterate through trials
    for trial in range(trials):
        opt_dir = location + 'Opt_info_' + name + '_' + str(trial+1) + '/'
        for dummy_count in range(1000):
            out_inner = opt_dir + 'optimize/' + str(dummy_count+1) + '/out~inner'
            try:
                os.remove(out_inner)
            except:
                break

if __name__ == '__main__':
    # Parsing input arguments
    parser = arg.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help='name of test to move data for')
    parser.add_argument("-l", "--location", required=True, help='directory of heron files where working directories are')
    parser.add_argument("-t", "--trials", required=True, help='number of trials for which folders are available')
    args = parser.parse_args()
    clear_outer_inners(args.name, args.location, int(args.trials))