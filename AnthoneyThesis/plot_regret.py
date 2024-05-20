import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections
import seaborn as sns

def determine_gold_value(csv_dir, opt_list, sample_count, eval_count, objective, decision_vars, best_count, max=True):
    """
        Determines the gold value to use for regret calculation based on test results
        @ In, csv_dir, directory pointing to csvs for given comparison
        @ In, opt_list, list of upper directories for csvs
        @ In, sample_count, number of samples for analysis
        @ In, eval_count, number of evaluations for each optimization (same for all csvs please)
        @ In, objective, name of objective of optimization
        @ In, decision_vars, names of decision variables for optimization
        @ In, best_count, number of samples to store and calculate gold value variance with
        @ In, max, bool that says whether it was a minimization or maximization problem
        @ Out, gold_value, recommended gold value
        @ Out, gold_dict, dictionary with detailed info from gold value determination run
    """
    # Initialize gold data dictionary used for comparison
    gold_dict = {}
    for method in opt_list:
        gold_dict.update({method:{}})

    # Initialize gold value
    if max:
        gold_value = -1e99
    else:
        gold_value = 1e99
    gold_method = None
    gold_decision = None
    
    # Iterating through each method and pulling the top best_count solutions ordering them accordingly
    for method in opt_list:
        # Initialize arrays for objectives and decisions
        if max:
            obj_array = -1e99*np.ones(best_count)
            obj_val = -1e99
        else:
            obj_array = 1e99*np.ones(best_count)
            obj_val = -1e99
        
        # Setting an array for each decision variable
        var_dict = {}
        for var in decision_vars:
            var_dict[var] = np.empty(best_count)
            var_dict[var+'_best'] = None
        
        # Tracking which sample had the best solution
        best_sample = 0
        
        # Iterating through samples for this method
        for samp in range(sample_count):
            # Load data
            csv = csv_dir + '/' + method + '/' + f'Opt_{samp+1}.csv'
            loaded_csv = pd.read_csv(csv)

            # Check csv
            finalized = np.asarray(loaded_csv['accepted'])[-1]
            if finalized != 'final':
                print('CSV is not complete')
                exit()

            # Retrieve final values
            sol_obj = np.asarray(loaded_csv[objective])[-1]
            temp_var_dict = {}
            for var in decision_vars:
                value = np.asarray(loaded_csv[var])[-1]
                temp_var_dict[var] = value
            
            # Compare against current list of solutions
            if max:
                worst_max = np.min(obj_array)
                worst_index = np.argmin(obj_array)
                if sol_obj > obj_val:
                    best_sample = samp+1
                    obj_val = sol_obj
                    obj_array[worst_index] = sol_obj
                    for var in decision_vars:
                        var_dict[var+'_best'] = temp_var_dict[var]
                        var_dict[var][worst_index] = temp_var_dict[var]
                elif sol_obj > worst_max:
                    obj_array[worst_index] = sol_obj
                    for var in decision_vars:
                        var_dict[var][worst_index] = temp_var_dict[var]
            else:
                worst_min = np.max(obj_array)
                worst_index = np.argmax(obj_array)
                if sol_obj < obj_val:
                    best_sample = samp+1
                    obj_val = sol_obj
                    obj_array[worst_index] = sol_obj
                    for var in decision_vars:
                        var_dict[var+'_best'] = temp_var_dict[var]
                        var_dict[var][worst_index] = temp_var_dict[var]
                elif sol_obj < worst_min:
                    obj_array[worst_index] = sol_obj
                    for var in decision_vars:
                        var_dict[var][worst_index] = temp_var_dict[var]
        # Print out some statistics from gold value determination for this method
        print(f'Displaying results for the optimization method {method}...\n')
        print(f'The estimated solution value is {obj_val}\n'
              f'The top {best_count} solution values are...\n {obj_array}\n'
              f'The variance of these is {np.var(obj_array)}\n'
              f'The average of these is {np.mean(obj_array)}'
              f'The percent std is {np.divide(np.std(obj_array),np.mean(obj_array))}')

        for var in decision_vars:
            print(f'The value of {var} at the best solution is {var_dict[var+"_best"]}\n'
                  f'The values of {var} at the top {best_count} solutions are...\n {var_dict[var]}\n'
                  f'The variance of these are {np.var(var_dict[var])}\n'
                  f'The average of these is {np.mean(var_dict[var])}'
                  f'The percent std is {np.divide(np.std(var_dict[var]),np.mean(var_dict[var]))}')
            
        # Storing total dictionary of information over all optimizers
        gold_dict[method].update(var_dict)
        gold_dict[method].update({'Objective Array':obj_array})
        gold_dict[method].update({'Objective Best':obj_val})
        gold_dict[method].update({'Best Sample':best_sample})

        if max:
            if obj_val > gold_value or gold_value is None:
                gold_value = obj_val
                gold_method = method
                gold_decision = {}
                for var in decision_vars:
                    gold_decision[var] = var_dict[var+'_best']
        else:
            if obj_val < gold_value or gold_value is None:
                gold_value = obj_val
                gold_method = method
                gold_decision = {}
                for var in decision_vars:
                    gold_decision[var] = var_dict[var+'_best']
    print(f'Finished skimming the results for a gold value...\n'
          f'The gold value is {gold_value}\n'
          f'The method that found this value is {gold_method}\n'
          f'The decision variables are...\n{gold_decision}\n')
    return gold_value, gold_dict

def generate_regret(csv_dir, opt_list, sample_count, eval_count, objective, gold_value):
    """
        Plot the regret for the results for a given TEA comparison
        @ In, csv_dir, directory pointing to csvs for given comparison
        @ In, opt_list, list of upper directories for csvs
        @ In, sample_count, number of samples for analysis
        @ In, eval_count, number of evaluations for each optimization (same for all csvs please)
        @ In, objective, name of objective of optimization
        @ In, gold_value, value of the true solution 
        @ Out, plotting_data, calculated regret curves for each optimizer and trial
    """
    # Dictionary to store data sets
    plotting_data_simple = {}
    plotting_data_cum = {}
    plotting_data_ave = {}
    # Iterating through each set of csvs (each method for analysis)
    for method in opt_list:
        # TODO Initialize data arrays for computation and plotting here
        simple_array = np.empty((sample_count, eval_count))
        cum_array = np.empty((sample_count, eval_count))
        ave_array = np.empty((sample_count, eval_count))
        for samp in range(sample_count):
            csv = csv_dir + '/' + method + '/' + f'Opt_{samp+1}.csv'
            loaded_csv = pd.read_csv(csv)
            eval_data = np.asarray(loaded_csv[objective][0:eval_count])
            # Solver solution at a given eval count, extra care for GD
            try:
                solution_data = np.asarray(loaded_csv['solutionValue'][0:eval_count])
            except:
                accepted_track = list(loaded_csv['accepted'][0:eval_count])
                index_list = []
                for accept_index in range(len(accepted_track)):
                    if accepted_track[accept_index] in ['first', 'accepted']:
                        index_list.append(accept_index)
                solution_data = np.empty(eval_count)
                # Need to push last accepted to end of optimization traj
                index_list.append(eval_count)
                for ind in range(len(index_list)-1):
                    solution_data[index_list[ind]:index_list[ind+1]] = eval_data[index_list[ind]]

            # Calculating regret for a given trial run and method
            simple_regret = calculate_simple(solution_data, gold_value)
            cumulative_regret = calculate_cumulative(eval_data, gold_value)
            average_regret = calculate_average(cumulative_regret)
            simple_array[samp,:] = simple_regret
            cum_array[samp,:] = cumulative_regret
            ave_array[samp,:] = average_regret
        plotting_data_simple.update({method:simple_array})
        plotting_data_cum.update({method:cum_array})
        plotting_data_ave.update({method:ave_array})
    return {'Simple':plotting_data_simple, 'Cumulative':plotting_data_cum, 'Average':plotting_data_ave}

def calculate_simple(solution_data, gold_value):
    """
        Uses a given data set and gold value to compute the simple regret
        @ In, solution_data, array, list of optimizer solution values at each evaluation count
        @ In, gold_value, float, trueish solution to the optmization problem
        @ Out, simple_regret, array, list of normalized simple regrets at each evaluation count
    """
    simple_regret = np.abs(np.divide(np.subtract(solution_data, gold_value), gold_value))
    return simple_regret

def calculate_cumulative(eval_data, gold_value):
    """
        Uses a given data set and gold value to compute the simple regret
        @ In, eval_data, array, list of optimizer evaluation values 
        @ In, gold_value, float, trueish solution to the optmization problem
        @ Out, cumulative_regret, array, list of normalized cumulative regret at each evaluation count
    """
    cum_regret = 0
    cumulative_regret = np.empty(len(eval_data))
    for eval_index, eval in enumerate(eval_data):
        cum_regret += np.abs(np.divide(np.subtract(eval, gold_value),gold_value))
        cumulative_regret[eval_index] = cum_regret
    return cumulative_regret

def calculate_average(cumulative_regret):
    """
        Calculates average regret
        @ In, cumulative_regret, array, list of normalized cumulative regret at each evaluation count
        @ Out, average_regret, array, list of normalized average regret at each evaluation count
    """
    average_regret = np.empty(len(cumulative_regret))
    for index, cum in enumerate(cumulative_regret):
        average_regret[index] = cum/(index+1)
    return average_regret

def determine_confidence(data, ratio):
    """
        Provides arrays for the correct ratio on upper and lower values of the data at each evaluation
        @ In, data, 2d array, regret for each trial at each evaluation
        @ In, ratio, float, percentage of trials to include within envelope
        @ Out, lower, array, array of lower bound values
        @ Out, upper, array, array of upper bound values
    """
    # Initialize upper and lower
    lower = np.empty(len(data[0,:]))
    upper = np.empty(len(data[0,:]))

    # Can't do exact ratios if trial count does not allow, so round up to include more data than less
    separation_count = np.rint(ratio*len(data[:,0]))
    true_ratio = separation_count/len(data[:,0])
    print(f'The desired ratio was {ratio}; however, the data allows for a true ratio of {true_ratio}')
    remove_count = len(data[:,0]) - separation_count

    # Checking if there is an even number of points to remove
    if remove_count % 2 == 0:
        upper_snip = int(remove_count/2)
        lower_snip = int(remove_count/2)
    # If odd, remove one more from the upper piece, since these are more likely to be outliers
    else:
        upper_snip = int(np.rint((remove_count/2)+0.1))
        lower_snip = int(upper_snip - 1)

    # Looping over each evaluation
    for eval in range(len(data[0,:])):
        # Let's look at all data for this evaluation (all trials)
        trial_data = data[:,eval]
        # Sort in ascending order
        sorted = np.sort(trial_data)
        upper[eval] = sorted[len(data[:,0])-1-upper_snip]
        lower[eval] = sorted[-1*(len(data[:,0])-lower_snip)]
    
    return lower, upper

def plot_simple(plotting_data, plotting_info, eval_count, data_marking={}, confidence='ratio'):
    """
        Uses stacked simple regret to plot mean and confidence intervals
        @ In, plotting_data, data to plot for simple regret
        @ In, plotting_info, dictionary of values for plotting
        @ In, eval_count, number of evaluations
        @ In, data_marking, dict, evaluation counts to plot actual data points on
        @ In, confidence, str, 'ratio' or 'std' to determine how to draw confidence bounds
    """
    plt.rcParams['text.usetex'] = True
    # Iteration vector
    iter_vec = np.linspace(1, eval_count, eval_count)
    method_count = 0
    ylim = 0
    # Iterate through plotting_data and plot each method
    for method, data in plotting_data.items():
        # Are we plotting specific regret values?
        if method in list(data_marking):
            for eval_counts in data_marking[method]:
                marks = data[:, eval_counts-1]
                plt.scatter(eval_counts*np.ones(len(marks)), marks, 
                            c=plotting_info['color_bank'][method_count], marker='x')
                ylim_new = np.max(marks)
                if ylim_new > ylim:
                    ylim = ylim_new
        plt.plot(iter_vec, np.mean(data, axis=0),
                 label=plotting_info['map'][method],
                 linewidth=1.5,
                 color=plotting_info['color_bank'][method_count]
                 )
        if confidence == 'ratio':
            lower_bound, upper_bound = determine_confidence(data, plotting_info['ratio'])
            plt.fill_between(iter_vec, lower_bound, upper_bound,
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, lower_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, upper_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(upper_bound)
        else:
            plt.fill_between(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)))
        method_count += 1
        if ylim_new > ylim:
            ylim = ylim_new
    plt.legend(fontsize=plotting_info['legend_font'], loc='upper right')
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_s(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.ylim(0, 1.1*ylim)
    plt.xlim(1, eval_count)
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.show()
    
def plot_cum(plotting_data, plotting_info, eval_count, data_marking={}, confidence='ratio'):
    """
        Uses stacked simple regret to plot mean and confidence intervals
        @ In, plotting_data, data to plot for cumulative regret
        @ In, plotting_info, dictionary of values for plotting
        @ In, eval_count, number of evaluations
        @ In, data_marking, dict, for each method what points to include as manually plotted
        @ In, confidence, str, what style of confidence bound to use
    """
    plt.rcParams['text.usetex'] = True
    # Iteration vector
    iter_vec = np.linspace(1, eval_count, eval_count)
    method_count = 0
    ylim = 0
    # Iterate through plotting_data and plot each method
    for method, data in plotting_data.items():
        # Are we plotting specific regret values?
        if method in list(data_marking):
            for eval_counts in data_marking[method]:
                marks = data[:, eval_counts-1]
                plt.scatter(eval_counts*np.ones(len(marks)), marks, 
                            c=plotting_info['color_bank'][method_count], marker='x')
                ylim_new = np.max(marks)
                if ylim_new > ylim:
                    ylim = ylim_new
        plt.plot(iter_vec, np.mean(data, axis=0),
                 label=plotting_info['map'][method],
                 linewidth=1.5,
                 color=plotting_info['color_bank'][method_count]
                 )
        if confidence == 'ratio':
            lower_bound, upper_bound = determine_confidence(data, plotting_info['ratio'])
            plt.fill_between(iter_vec, lower_bound, upper_bound,
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, lower_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, upper_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(upper_bound)
        else:
            plt.fill_between(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)))
        method_count += 1
        if ylim_new > ylim:
            ylim = ylim_new
    plt.legend(fontsize=plotting_info['legend_font'], loc='upper right')
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_c(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.ylim(0, 1.1*ylim)
    plt.xlim(1, eval_count)
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.show()
    
def plot_ave(plotting_data, plotting_info, eval_count, data_marking={}, confidence='ratio'):
    """
        Uses stacked simple regret to plot mean and confidence intervals
        @ In, plotting_data, data to plot for average regret
        @ In, plotting_info, dictionary of values for plotting
        @ In, eval_count, number of evaluations
        @ In, data_marking, dict, for each method what evals to plot trial data for
        @ In, confidence, str, ratio or std for confidence bounds
    """
    plt.rcParams['text.usetex'] = True
    # Iteration vector
    iter_vec = np.linspace(1, eval_count, eval_count)
    method_count = 0
    ylim = 0
    # Iterate through plotting_data and plot each method
    for method, data in plotting_data.items():
        # Are we plotting specific regret values?
        if method in list(data_marking):
            for eval_counts in data_marking[method]:
                marks = data[:, eval_counts-1]
                plt.scatter(eval_counts*np.ones(len(marks)), marks, 
                            c=plotting_info['color_bank'][method_count], marker='x')
                ylim_new = np.max(marks)
                if ylim_new > ylim:
                    ylim = ylim_new
        plt.plot(iter_vec, np.mean(data, axis=0),
                 label=plotting_info['map'][method],
                 linewidth=1.5,
                 color=plotting_info['color_bank'][method_count]
                 )
        if confidence == 'ratio':
            lower_bound, upper_bound = determine_confidence(data, plotting_info['ratio'])
            plt.fill_between(iter_vec, lower_bound, upper_bound,
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, lower_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, upper_bound,
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(upper_bound)
        else:
            plt.fill_between(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                            color=plotting_info['color_bank'][method_count],
                            alpha=0.1
                            )
            plt.plot(iter_vec, np.subtract(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            plt.plot(iter_vec, np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)),
                    color=plotting_info['color_bank'][method_count],
                    linestyle='--', linewidth=1)
            ylim_new = np.max(np.add(np.mean(data, axis=0), plotting_info['std']*np.std(data, axis=0)))
        method_count += 1
        if ylim_new > ylim:
            ylim = ylim_new
    plt.legend(fontsize=plotting_info['legend_font'], loc='upper right')
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_a(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.ylim(0, 1.1*ylim)
    plt.xlim(1, eval_count)
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.show()

def regret_histogram(hist_data, hist_info, dist='normal'):
    """
        Plots histogram and normal distribution with estimated 
        mean and variance from data to compare fit
        @ In, hist_data, np.array, values used to generate histogram
        @ In, hist_info, dict, plotting info for the histogram
    """
    mu = np.mean(hist_data)
    var = np.var(hist_data)
    plt.hist(hist_data, density=True, bins=hist_info['bins'], color=hist_info['color'], alpha=hist_info['alpha'])
    plt.xlabel(hist_info['x-axis'], fontsize=hist_info['x size'])
    plt.ylabel(hist_info['y-axis'], fontsize=hist_info['y size'])
    plt.title(hist_info['title'], fontsize=hist_info['title size'])
    dist_estimates = distribution_estimate(mu, var, dist, hist_info['sample_points'])
    plt.plot(hist_info['sample_points'], dist_estimates, color=hist_info['color'], linestyle='--', linewidth=1.75)
    plt.show()

def distribution_estimate(mean, var, dist, sample_points):
    """
        Generates curve estimate of desired dist, using data mean and var
        @ In, mean, sample mean of data
        @ In, var, sample variance of data
        @ In, dist, type of distribution fit
        @ In, sample_points, np.array, set of points to evaluate the distribution estimate at
        @ Out, dist_estimate, np.array of distribution estimates
    """
    # The normal distribution
    if dist == 'normal':
        exp_arg1 = np.subtract(sample_points, mean)
        exp_arg2 = np.square(exp_arg1)
        exp_arg3 = np.multiply(-(1/(2*var)), exp_arg2)
        return np.multiply(1/(np.sqrt(np.pi*2*var)), np.exp(exp_arg3))
    else:
        print(f'That distribution is not offered, dumbo')
        exit()

def plot_trajectories(plotting_data, plotting_info):
    """
        Uses regret data to plot all trajectories and mean value
        @ In, plotting_data, data to plot for all regrets
        @ In, plotting_info, dictionary of values for plotting
    """
    # Pulling and separating data dicts
    simple_data = plotting_data['Simple']
    cum_data = plotting_data['Cumulative']
    ave_data = plotting_data['Average']

    # Pulling any method to get eval_vec for plotting
    first_method = list(plotting_info['map'])[0]
    eval_vec = np.linspace(1, len(simple_data[first_method][0,:]), len(simple_data[first_method][0,:]))
    color_index = 0
    ylim = 0

    # Start with simple regret
    for method, data in simple_data.items():
        # Let's plot the mean trajectory first
        plt.plot(eval_vec, np.mean(data, axis=0),
                 color=plotting_info['color_bank'][color_index],
                 linewidth=2,
                 marker='.', markersize=9,
                 label=plotting_info['map'][method])
        # Assumes that there is at least one trial
        for trial in range(len(data[:,0])):
            trajectory = data[trial,:]
            plt.plot(eval_vec, trajectory,
                     color=plotting_info['color_bank'][color_index],
                     linewidth=1, alpha = plotting_info['alpha'])
            ylim_new = 1.1*np.max(trajectory)
            if ylim_new > ylim:
                ylim = ylim_new
        color_index += 1
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_s(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.xlim([1,len(simple_data[first_method][0,:])])
    plt.ylim([0,ylim])
    plt.legend(fontsize=plotting_info['legend_font'])
    plt.show()

    # Now cumulative regret
    color_index = 0
    ylim = 0
    for method, data in cum_data.items():
        # Let's plot the mean trajectory first
        plt.plot(eval_vec, np.mean(data, axis=0),
                 color=plotting_info['color_bank'][color_index],
                 linewidth=2,
                 marker='.', markersize=9,
                 label=plotting_info['map'][method])
        # Assumes that there is at least one trial
        for trial in range(len(data[:,0])):
            trajectory = data[trial,:]
            plt.plot(eval_vec, trajectory,
                     color=plotting_info['color_bank'][color_index],
                     linewidth=1, alpha = plotting_info['alpha'])
            ylim_new = 1.1*np.max(trajectory)
            if ylim_new > ylim:
                ylim = ylim_new
        color_index += 1
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_c(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.xlim([1,len(simple_data[first_method][0,:])])
    plt.ylim([0,ylim])
    plt.legend(fontsize=plotting_info['legend_font'])
    plt.show()

    # Now average regret
    color_index = 0
    ylim = 0
    for method, data in ave_data.items():
        # Let's plot the mean trajectory first
        plt.plot(eval_vec, np.mean(data, axis=0),
                 color=plotting_info['color_bank'][color_index],
                 linewidth=2,
                 marker='.', markersize=9,
                 label=plotting_info['map'][method])
        # Assumes that there is at least one trial
        for trial in range(len(data[:,0])):
            trajectory = data[trial,:]
            plt.plot(eval_vec, trajectory,
                     color=plotting_info['color_bank'][color_index],
                     linewidth=1, alpha = plotting_info['alpha'])
            ylim_new = 1.1*np.max(trajectory)
            if ylim_new > ylim:
                ylim = ylim_new
        color_index += 1
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_a(\\tau)$', fontsize=plotting_info['axis_font'])
    plt.title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.xlim([1,len(simple_data[first_method][0,:])])
    plt.ylim([0,ylim])
    plt.legend(fontsize=plotting_info['legend_font'])
    plt.show()

def plot_violins(plotting_data, plotting_info, eval_indices=np.array([1,10,30,50]), inner='stick', width=0.75, delta=0.03, split=True, save_loc=None, palette=None):
    """
        Uses regret data to plot violins of computer experiments
        @ In, plotting_data, data to plot for all regrets
        @ In, plotting_info, dictionary of values for plotting  
        @ In, eval_indices, np.array, tells what evals to make violins for
        @ In, inner, str, 'stick' or None
        @ In, width, float, width of plot
        @ In, delta, float, width of gap between split
        @ In, split, bool, whether to split violins or not
        @ In, save_loc, str, directory to save pngs
        @ In, palette, str, name of color palette to use
    """
    # Plotting violins with swarms to represent distributed data
    plot_data_simple = plotting_data['Simple']
    plot_data_cum = plotting_data['Cumulative']
    plot_data_ave = plotting_data['Average']
    eval_columns = eval_indices.astype(str).tolist()

    # Let's track some lengths
    meth_count = len(list(plot_data_simple))
    trial_count = len(plot_data_simple[list(plot_data_simple)[0]][:,0])

    # Lists that will form the data frame dictionary
    name_list = []
    eval_list = []
    trial_list = []
    simple_list = []
    cum_list = []
    ave_list = []

    # Making dict for data frame for violin plots
    for method in list(plot_data_simple):
        # Extending name directory
        name_list.extend([plotting_info['map'][method]]*(trial_count*len(eval_columns)))
        # Looping over evaluation counts to plot data for
        for eval_amt in eval_columns:
            # Time extend eval list
            eval_list.extend([eval_amt]*trial_count)
            simple_data = plot_data_simple[method][:,int(eval_amt)-1]
            cum_data = plot_data_cum[method][:,int(eval_amt)-1]
            ave_data = plot_data_ave[method][:,int(eval_amt)-1]
            # Iterating through each trial
            for trial in range(trial_count):
                # Extend another list
                trial_list.extend([trial+1])
                simple_list.extend([simple_data[trial]])
                cum_list.extend([cum_data[trial]])
                ave_list.extend([ave_data[trial]])
    # Want arrays for actual data
    simple_array = np.array(simple_list)
    cum_array = np.array(cum_list)
    ave_array = np.array(ave_list)

    # Build dictionary to construct data frame from
    df_dict = {'Name':name_list,
                'Evaluations':eval_list,
                'Trial':trial_list,
                'Simple Regret':simple_array,
                'Cumulative Regret':cum_array,
                'Average Regret':ave_array}
    df = pd.DataFrame(df_dict)
    if palette is None:
        color_block = plotting_info['color_bank'][0:len(list(plotting_data))-1]
    else:
        color_block = palette

    simp_ax = sns.catplot(data=df, x='Evaluations', y='Simple Regret', hue='Name', legend=False, linecolor='black', saturation=1, height=8.27, aspect=11.7/8.27,
                kind="violin", inner=inner, split=split, palette=color_block, density_norm='area', inner_kws=dict(color='black'))
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_s(\\tau)$', fontsize=plotting_info['axis_font'])
    simp_ax.figure.get_axes()[0].legend(loc='best', title=None, fontsize=plotting_info['legend_font'])
    simp_ax.figure.get_axes()[0].set_title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.ylim(bottom=0)
    if split:
        final_width = width - delta
        offset_violinplot_halves(simp_ax, delta, final_width, inner, 'vertical')
    if save_loc is not None:
        plt.savefig(save_loc+'/'+plotting_info['name']+'_simple_violin.png', bbox_inches="tight")

    cum_ax = sns.catplot(data=df, x='Evaluations', y='Cumulative Regret', hue='Name', legend=False, linecolor='black', saturation=1, height=8.27, aspect=11.7/8.27,
                kind="violin", inner=inner, split=split, palette=color_block, density_norm='area', inner_kws=dict(color='black'))
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_c(\\tau)$', fontsize=plotting_info['axis_font'])
    cum_ax.figure.get_axes()[0].legend(loc='best', title=None, fontsize=plotting_info['legend_font'])
    cum_ax.figure.get_axes()[0].set_title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.ylim(bottom=0)
    if split:
        final_width = width - delta
        offset_violinplot_halves(cum_ax, delta, final_width, inner, 'vertical')
    if save_loc is not None:
        plt.savefig(save_loc+'/'+plotting_info['name']+'_cumu_violin.png', bbox_inches="tight")

    ave_ax = sns.catplot(data=df, x='Evaluations', y='Average Regret', hue='Name', legend=False, linecolor='black', saturation=1, height=8.27, aspect=11.7/8.27,
                kind="violin", inner=inner, split=split, palette=color_block, density_norm='area', inner_kws=dict(color='black'))
    plt.xlabel('$\\tau$', fontsize=plotting_info['axis_font'])
    plt.ylabel('$R_a(\\tau)$', fontsize=plotting_info['axis_font'])
    ave_ax.figure.get_axes()[0].legend(loc='best', title=None, fontsize=plotting_info['legend_font'])
    ave_ax.figure.get_axes()[0].set_title(plotting_info['title'], fontsize=plotting_info['title_font'])
    plt.ylim(bottom=0)
    if split:
        final_width = width - delta
        offset_violinplot_halves(ave_ax, delta, final_width, inner, 'vertical')
    if save_loc is not None:
        plt.savefig(save_loc+'/'+plotting_info['name']+'_ave_violin.png', bbox_inches="tight")

def offset_violinplot_halves(ax, delta, width, inner, direction):
    """
    This function offsets the halves of a violinplot to compare tails
    or to plot something else in between them. This is specifically designed
    for violinplots by Seaborn that use the option `split=True`.

    For lines, this works on the assumption that Seaborn plots everything with
     integers as the center.

    Args:
     <ax>    The axis that contains the violinplots.
     <delta> The amount of space to put between the two halves of the violinplot
     <width> The total width of the violinplot, as passed to sns.violinplot()
     <inner> The type of inner in the seaborn
     <direction> Orientation of violinplot. 'hotizontal' or 'vertical'.

    Returns:
     - NA, modifies the <ax> directly
    """
    # offset stuff
    if inner == 'stick':
        lines = ax.figure.get_axes()[0].get_lines()
        for line in lines:
            if direction == 'horizontal':
                data = line.get_ydata()
                if int(data[0] + 1)/int(data[1] + 1) < 1:
                    # type is top, move neg, direction backwards for horizontal
                    data -= delta
                else:
                    # type is bottom, move pos, direction backward for hori
                    data += delta
                line.set_ydata(data)
            elif direction == 'vertical':
                data = line.get_xdata()
                if int(data[0] + 1)/int(data[1] + 1) < 1:
                    # type is left, move neg
                    data -= delta
                else:
                    # type is left, move pos
                    data += delta
                line.set_xdata(data)


    for ii, item in enumerate(ax.figure.get_axes()[0].collections):
        # axis contains PolyCollections and PathCollections
        if isinstance(item, matplotlib.collections.PolyCollection):
            # get path
            path, = item.get_paths()
            vertices = path.vertices
            half_type = _wedge_dir(vertices, direction)
            # shift x-coordinates of path
            if half_type in ['top','bottom']:
               if inner in ["stick", None]:
                    if half_type == 'top': # -> up
                        vertices[:,1] -= delta
                    elif half_type == 'bottom': # -> down
                        vertices[:,1] += delta
            elif half_type in ['left', 'right']:
                if inner in ["stick", None]:
                    if half_type == 'left': # -> left
                        vertices[:,0] -= delta
                    elif half_type == 'right': # -> down
                        vertices[:,0] += delta

def _wedge_dir(vertices, direction):
    """
    Args:
      <vertices>  The vertices from matplotlib.collections.PolyCollection
      <direction> Direction must be 'horizontal' or 'vertical' according to how
                   your plot is laid out.
    Returns:
      - a string in ['top', 'bottom', 'left', 'right'] that determines where the
         half of the violinplot is relative to the center.
    """
    if direction == 'horizontal':
        result = (direction, len(set(vertices[1:5,1])) == 1)
    elif direction == 'vertical':
        result = (direction, len(set(vertices[-3:-1,0])) == 1)
    outcome_key = {('horizontal', True): 'bottom',
                   ('horizontal', False): 'top',
                   ('vertical', True): 'left',
                   ('vertical', False): 'right'}
    # if the first couple x/y values after the start are the same, it
    #  is the input direction. If not, it is the opposite
    return outcome_key[result]

if __name__ == '__main__':
    csv_dir = 'ies_csv'
    # csv_dir = 'optimization_settings_csv2'
    opt_list = ['GD','BO']
    sample_count = 50
    eval_count = 50
    # sample_count = 100
    # eval_count = 15
    objective = 'mean_NPV'
    decision_vars = ['H2_storage_capacity', 'HTSE_capacity', 'wind_capacity', 'npp_capacity']
    best_count = 10
    max = True
    # determine_gold_value(csv_dir, opt_list, sample_count, eval_count, objective, decision_vars, best_count, max)
    gold_value = -2.31252655e8
    # gold_value = 80.22
    plotting_info = {'map':{'GD':'Gradient Descent','BO':'Bayesian Optimization'},
                     'color_bank':['blue', 'red', 'green', 'black', 'maroon', 'orange'],
                     'axis_font':16,
                     'title_font':18,
                     'legend_font':14,
                     'title':'IES Workshop Test',
                     'name':'IES',
                     'alpha':0.3,
                     'std':3,
                     'ratio':0.95}
    # Decides of specific regret values are marked for clarity
    # evals = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # evals = [10]
    # data_marking = {'BO':[1, 2, 6], 'GD':[4, 8]}
    data_marking = {}
    confidence = 'ratio'
    plot_data = generate_regret(csv_dir, opt_list, sample_count, eval_count, objective, gold_value)
    # plot_simple(plot_data['Simple'], plotting_info, eval_count,data_marking, confidence)
    # plot_cum(plot_data['Cumulative'], plotting_info, eval_count, data_marking, confidence)
    # plot_ave(plot_data['Average'], plotting_info, eval_count, data_marking, confidence)
    # exit()
    # Let's look at hist data for each regret
    # hist_data_simple = plot_data['Simple']['BO'][:,9]
    # hist_data_cum = plot_data['Cumulative']['BO'][:,9]
    # hist_data_simple = plot_data['Average']['BO'][:,9]
    # hist_info = {'bins':20,
    #              'color':'red',
    #              'title':'BO Simple Regret (Evaluation=10)',
    #              'title size': 24,
    #              'x-axis':'Simple Regret',
    #              'x size':18,
    #              'y-axis':'Probability',
    #              'y size':18,
    #              'alpha':0.25,
    #              'sample_points':np.linspace(0, 1.2*np.max(hist_data_simple),100),
    #              }
    # regret_histogram(hist_data_simple, hist_info)
    # plot_trajectories(plot_data, plotting_info)
    plot_violins(plot_data, plotting_info, save_loc='../../Desktop', palette='pastel')
