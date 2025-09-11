# GENETIC_ALGORITHM.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses genetic algorithm (from scratch) to optimize (maximize) an erratic function with many sharp peaks and several 'distracter' answers
# Objective function is similar to the function here https://mathblag.wordpress.com/2013/09/01/sums-of-periodic-functions/

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(123456)

#--------------#
#  PARAMETERS  #
#--------------#

total_pop = 500       # population size (invariant across generations)
generations = 100     # total iterations
mutation_rate = 0.05  # frequency of random mutations in the population
x_lower = 0           # lower bound of parameter space (x-value)
x_upper = 400         # upper bound of parameter space (x-value)

#----------------------#
#  OBJECTIVE FUNCTION  #
#----------------------#

def obj_func(x_pt):
    # y(x) = | sin(a*x) + sin(b*x) + sin(c*x) |
    # a = 2*pi/13
    # b = 2*pi/18
    # c = 2*pi/23
    term1 = np.sin(2 * math.pi * x_pt / 13)
    term2 = np.sin(2 * math.pi * x_pt / 18)
    term3 = np.sin(2 * math.pi * x_pt / 23)
    y_pt = abs(term1 + term2 + term3)
    return y_pt

#-------------#
#  ALGORITHM  #
#-------------#

person_numbers = list(range(total_pop))  # identifiers for the synthetic people in the population

x_pop = list(np.random.uniform(low=x_lower, high=x_upper, size=total_pop))  # randomly selects x-values for the initial population
global_best_y = -math.inf  # initialize to be worst possible value (negative infinity - because GREATER values are sought)

# fitness evaluation for initial population
y_pt = [obj_func(x) for x in x_pop]  # evaluate objective function for each person in current population
fitness = y_pt  # fitness is simply the y-value (maximize)
fitness = [f/sum(fitness) for f in fitness]  # renormalize fitness values to sum to 1 (corresponding to probabilities)

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
        
        # mutation (replace some x-values with randomly selected values from within the parameter space, using the mutation rate)
        if np.random.uniform(low=0, high=1) < mutation_rate:
            x_value_new = np.random.uniform(low=x_lower, high=x_upper)
        
        # save
        x_new.append(x_value_new)

    # update quantities
    x_pop = x_new
    y_pt = [obj_func(x) for x in x_pop]
    fitness = y_pt
    fitness = [f/sum(fitness) for f in fitness]  # renormalize to sum to one

    # compare current population to running global optimal solution
    if max(y_pt) > global_best_y:
        global_best_y = max(y_pt)
        person_num = y_pt.index(global_best_y)  # determine person in current generation with the best objective function value
        global_best_x = x_pop[person_num]

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 10
axis_labels_size = 8
axis_ticks_size = 7
legend_size = 6
line_width = 1
star_size = 30

buffer_plot = 1.2     # space atop for label - first plot (one = no buffer)
buffer_point = 0.002  # space around optimal point - second plot (zero = no buffer)

# objective function (blue curve)
y_plot = []
x_plot = list(np.arange(x_lower, x_upper, 0.01))
for k in range(len(x_plot)):
    x_pt = x_plot[k]
    y_pt = obj_func(x_pt)
    y_plot.append(y_pt)

# best point label (red star)
X_best = str(round(global_best_x, 3))
Y_best = str(round(global_best_y, 3))
rate = str(100*mutation_rate)
best_pt_label = '(X=' + X_best + ', obj=' + Y_best + ')'

parameters_label = '(' + str(total_pop) + ' people, ' + str(generations) + ' generations, ' + rate + '% mutation)'

# plot limits
ymax1 = buffer_plot * max(y_plot)
xmin2 = (1 - buffer_point) * global_best_x
xmax2 = (1 + buffer_point) * global_best_x
ymin2 = (1 - buffer_point) * global_best_y
ymax2 = (1 + buffer_point) * global_best_y

mult = 0.975
xtext2 = xmin2 + mult*(xmax2 - xmin2)
ytext2 = ymin2 + mult*(ymax2 - ymin2)

# generate plots
for fig in [1,2]:
    # entire parameter space
    if fig == 1:
        fig1 = plt.figure(facecolor='lightblue')
        plt.gca().set_facecolor('white')
        plt.title('Genetic Algorithm', fontsize=title_size, fontweight='bold')
        plt.text(mult*x_upper, mult*ymax1, parameters_label, va='top', ha='right', fontsize=legend_size, zorder=30)
        plt.xlim(x_lower, x_upper)
        plt.ylim(0, ymax1)
    # close-up on optimal
    elif fig == 2:
        fig2 = plt.figure(facecolor='lightblue')
        plt.gca().set_facecolor('white')
        plt.title('Genetic Algorithm (zoomed-in)', fontsize=title_size, fontweight='bold')
        plt.text(xtext2, ytext2, parameters_label, va='top', ha='right', fontsize=legend_size, zorder=30)
        plt.xlim(xmin2, xmax2)
        plt.ylim(ymin2, ymax2)
    plt.plot(x_plot, y_plot, color='blue', linewidth=line_width, label='Objective function')
    plt.scatter(global_best_x, global_best_y, marker='*', s=star_size, color='red', label='Maximum ' + best_pt_label)
    plt.legend(loc='upper left', ncol=1, fontsize=legend_size, facecolor='whitesmoke', framealpha=1).set_zorder(15)
    plt.ylabel('Objective function', fontsize=axis_labels_size)
    plt.xlabel('Parameter value (X)', fontsize=axis_labels_size)
    plt.xticks(fontsize=axis_ticks_size)
    plt.yticks(fontsize=axis_ticks_size)
    plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
    plt.show(True)

#----------------#
#  EXPORT PLOTS  #
#----------------#

pdf = PdfPages('Genetic_algorithm_graphics.pdf')
fig_list = [fig1, fig2]
for fig in fig_list:
    f = fig_list.index(fig) + 1
    fig.savefig('GENA-figure-' + str(f) + '.jpg', dpi=300)
    pdf.savefig(fig)
pdf.close()
del pdf, fig, f

###

# runtime
runtime_sec = round(time.time()-time0, 2)
if runtime_sec < 60:
    print('\nruntime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec/60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\nruntime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0

