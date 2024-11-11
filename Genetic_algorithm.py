# GENETIC_ALGORITHM.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses a genetic algorithm (from scratch) to optimize (maximize) an erratic function with myriad sharp peaks and local extrema
# Objective function is similar to that presented here https://mathblag.wordpress.com/2013/09/01/sums-of-periodic-functions/

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(123456)
ver = ''  # version (empty or integer)

topic = 'Genetic algorithm'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

total_pop = 50         # population size (invariant across generations)
generations = 200      # total iterations
mutation_rate = 0.05   # frequency of random mutations in the population
x_lower = 0            # lower bound of parameter space (x-value)
x_upper = 400          # upper bound of parameter space (x-value)

#----------------------#
#  OBJECTIVE FUNCTION  #
#----------------------#

def obj_func(x_pt):
    # y(x) = | sin(a * x) + sin(b * x) + sin(c * x) |
    # a = (2 * pi) / 13
    # b = (2 * pi) / 18
    # c = (2 * pi) / 23
    term1 = np.sin(2 * math.pi * x_pt / 13)
    term2 = np.sin(2 * math.pi * x_pt / 18)
    term3 = np.sin(2 * math.pi * x_pt / 23)
    y_pt = abs(term1 + term2 + term3)
    return y_pt

#-------------#
#  ALGORITHM  #
#-------------#

person_numbers = list(range(total_pop))  # serial numbers (identifiers) for the synthetic people in the population

x_pop = list(np.random.uniform(low=x_lower, high=x_upper, size=total_pop))  # randomly selects x-values for the initial population
global_best_y = -math.inf  # initialize to be worst possible value (negative infinity - because GREATER values are sought)

# fitness evaluation for initial population
y_pt = [obj_func(x) for x in x_pop]  # evaluate objective function for each person in current population
fitness = y_pt  # fitness is simply the y-value (maximize)
fitness = [f / sum(fitness) for f in fitness]  # renormalizes so fitness values sum to 1 (corresponding to probabilities)

# breeding (each person has a single gene - their x-value)
for g in range(generations):
    x_new = []
    for i in range(total_pop):
        # selection (randomly select parents, in proportion to their fitness value, with replacement)
        parent_1 = np.random.choice(person_numbers, p=fitness)
        parent_2 = parent_1  # initially, set the two parents to be the same
        while parent_1 == parent_2:  # then iterate until they differ (so no persons are paired with themselves)
            parent_2 = np.random.choice(person_numbers, p=fitness)
        
        # crossover (randomly select x-value from one parent and assign it to the offspring)
        if np.random.uniform(low=0, high=1) < 0.5:
            x_value_new = x_pop[parent_1]
        else:
            x_value_new = x_pop[parent_2]
        
        # mutation (replace some x-values with randomly selected values from within the parameter space, according to the mutation rate)
        if np.random.uniform(low=0, high=1) < mutation_rate:
            x_value_new = np.random.uniform(low=x_lower, high=x_upper)
        
        # save
        x_new.append(x_value_new)

    # update quantities
    x_pop = x_new
    y_pt = [obj_func(x) for x in x_pop]
    fitness = y_pt
    fitness = [f / sum(fitness) for f in fitness]

    # compare current population to running global optimal solution
    if max(y_pt) > global_best_y:
        global_best_y = max(y_pt)
        person_num = y_pt.index(global_best_y)  # determine person in current generation with the best objective function value
        global_best_x = x_pop[person_num]

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 11
axis_labels_size = 8
axis_ticks_size = 8
legend_size = 8
line_width = 1.25
buffer_plot = 1.2     # space atop for label - objective function plot (one = no buffer)
buffer_point = 0.05   # space around optimal point - zoomed-in plot (zero = no buffer)

# objective function (blue curve)
y_plot = []
x_plot = list(np.arange(x_lower, x_upper, 0.01))
for k in range(len(x_plot)):
    x_pt = x_plot[k]
    y_pt = obj_func(x_pt)
    y_plot.append(y_pt)

best_pt_label = '(x = ' + str(round(global_best_x, 2)) + ', y = ' + str(round(global_best_y, 2)) + ')'  # point chosen (red star)

# generate plots
for fig in [1, 2]:
    # overall parameter space
    if fig == 1:
        fig1 = plt.figure()
        plt.title(topic + ' - objective function', fontsize=title_size, fontweight='bold')
        plt.xlim(x_lower, x_upper)
        plt.ylim(0, buffer_plot * max(y_plot))
    # close-up on maximum value
    elif fig == 2:
        fig2 = plt.figure()
        plt.title(topic + ' - maximum value (zoomed-in)', fontsize=title_size, fontweight='bold')
        plt.xlim((1 - buffer_point) * global_best_x, (1 + buffer_point) * global_best_x)
        plt.ylim((1 - buffer_point) * global_best_y, (1 + buffer_point) * global_best_y)
    plt.plot(x_plot, y_plot, color='blue', linewidth=line_width, label='objective function')
    plt.scatter(global_best_x, global_best_y, marker='*', s=50, color='red', label='maximum ' + best_pt_label)
    plt.legend(loc='upper left', ncol=2, fontsize=legend_size, facecolor='white', framealpha=1)
    plt.ylabel('Objective function (y)', fontsize=axis_labels_size)
    plt.xlabel('Parameter value (x)', fontsize=axis_labels_size)
    plt.xticks(fontsize=axis_labels_size)
    plt.yticks(fontsize=axis_labels_size)
    plt.grid(True, alpha=0.5, zorder=0)
    plt.show(True)

#----------#
#  EXPORT  #
#----------#

# export plots (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
for f in [fig1, fig2]:
    pdf.savefig(f)
pdf.close()
del pdf, f

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


