# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np

from shared_utils import *

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('axes', labelsize=22)
rc('xtick', labelsize=22)
rc('ytick', labelsize=22)
rc('legend', fontsize=13)

mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val


def colorscale(hexstr, scalefactor):
    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (int(r), int(g), int(b))


def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None, fmt='-o', label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.semilogx(x, y, fmt, color=color, label=label)
    ax.fill_between(x, np.clip(ymax, 0, 1), np.clip(ymin, 0, 1), color=color, alpha=alpha_fill)

linestyles = ['-', '-.', '--', ':']
brightness = [1.25, 1.0, 0.75, 0.5]
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
colors_old = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
colors = ['#2196f3', '#f44336', '#9c27b0', '#64dd17', '#009688', '#ff9800', '#795548', '#607d8b']

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# Define results path.
test_type = sys.argv[1]
path = './paper_results/'
path += test_type + '/'

# Define DR methods.
if test_type == 'multiv':
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value,
                     DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value,
                     DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
else:
    dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value,
                     DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value,
                     DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value,
                     DimensionalityReduction.BBSDh.value, DimensionalityReduction.Classif.value]

# Define number of random runs to average results over. 
rand_runs = 5

# Define significance level.
sign_level = 0.05

# Define standard sample counts.
if test_type == 'multiv':
    samples = [10, 20, 50, 100, 200, 500, 1000]
else:
    samples = [10, 20, 50, 100, 200, 500, 1000, 10000]

# Define which shifts we should include in our results.
shift_classes = ['small_gn_shift', 'medium_gn_shift', 'large_gn_shift', 'adversarial_shift', 'ko_shift',
                 'small_image_shift', 'medium_image_shift', 'large_image_shift', 'medium_img_shift+ko_shift',
                 'only_zero_shift+medium_img_shift']

# Define which datasets we should include in our results.
datasets = ['mnist', 'cifar10']

# -------------------------------------------------
# TABLE GENERATION
# -------------------------------------------------

# Define tables in which to store shift detection results in.
# dr_table      ...     shift detection accuracy per DR technique over all tested sample sizes and shifts (Table 1 a)).
# shift_table   ...     shift detection accuracy per shift over all tested sample sizes and DR techniques (Table 1 b)).
# 
# table[:,:,0]  = shift detection accuracy (i.e. table[:,:,1] / table[:,:,2]).
# table[:,:,1]  = shift detection count.
# table[:,:,2]  = total shift count.
dr_acc_table = np.zeros((len(dr_techniques), len(samples), 3))
shift_acc_table = np.zeros((len(shift_classes), len(samples), 3))

int_acc_table = np.zeros((3, len(samples), 3))
perc_acc_table = np.zeros((3, len(samples), 3))

dr_p_table = np.zeros((len(dr_techniques), len(samples), 3 * len(shift_classes) * rand_runs * 2))
shift_p_table = np.zeros((len(shift_classes), len(samples), 3 * len(dr_techniques) * rand_runs * 2))

p_table_ind = 0

for dataset in datasets:
    
    for shift_class_idx, shift_class in enumerate(shift_classes):
        
        # Reset sample sizes to default values.
        if test_type == 'multiv':
            samples = [10, 20, 50, 100, 200, 500, 1000]
        else:
            samples = [10, 20, 50, 100, 200, 500, 1000, 10000]
        
        # Define specific shifts per shift class.
        if shift_class == 'small_gn_shift':
            shifts = ['small_gn_shift_0.1',
                      'small_gn_shift_0.5', 'small_gn_shift_1.0']
        elif shift_class == 'medium_gn_shift':
            shifts = ['medium_gn_shift_0.1',
                      'medium_gn_shift_0.5',
                      'medium_gn_shift_1.0']
        elif shift_class == 'large_gn_shift':
            shifts = ['large_gn_shift_0.1',
                      'large_gn_shift_0.5',
                      'large_gn_shift_1.0']
        elif shift_class == 'adversarial_shift':
            shifts = ['adversarial_shift_0.1',
                      'adversarial_shift_0.5',
                      'adversarial_shift_1.0']
        elif shift_class == 'ko_shift':
            shifts = ['ko_shift_0.1',
                      'ko_shift_0.5',
                      'ko_shift_1.0']
        elif shift_class == 'orig':
            shifts = ['rand', 'orig']
        elif shift_class == 'small_image_shift':
            shifts = ['small_img_shift_0.1',
                      'small_img_shift_0.5',
                      'small_img_shift_1.0']
        elif shift_class == 'medium_image_shift':
            shifts = ['medium_img_shift_0.1',
                      'medium_img_shift_0.5',
                      'medium_img_shift_1.0']
        elif shift_class == 'large_image_shift':
            shifts = ['large_img_shift_0.1',
                      'large_img_shift_0.5',
                      'large_img_shift_1.0']
        elif shift_class == 'medium_img_shift+ko_shift':
            shifts = ['medium_img_shift_0.5+ko_shift_0.1',
                      'medium_img_shift_0.5+ko_shift_0.5',
                      'medium_img_shift_0.5+ko_shift_1.0']
        elif shift_class == 'only_zero_shift+medium_img_shift':
            shifts = ['only_zero_shift+medium_img_shift_0.1',
                      'only_zero_shift+medium_img_shift_0.5',
                      'only_zero_shift+medium_img_shift_1.0']
            samples = [10, 20, 50, 100, 200, 500, 1000]
        else:
            shifts = []
            print('Shift not recognized!')
    
        # Construct path to directory holding p-values from experiments.
        local_path = path + dataset + '_' + shift_class + '/'
        
        # Load p-values from shift class experiments.
        # Recall: samples_shifts_rands_dr_tech.shape = (len(samples), len(shifts), random_runs, len(dr_techniques))
        samples_shifts_rands_dr_tech = np.load(local_path + 'samples_shifts_rands_dr_tech.npy')
        shift_decision_array = samples_shifts_rands_dr_tech.copy()

        # for i in range(samples_shifts_rands_dr_tech.shape[1]):
        #     for j in range(samples_shifts_rands_dr_tech.shape[2]):
        #         dr_p_table[:,:,p_table_ind] = np.transpose(samples_shifts_rands_dr_tech[:,i,j,:])
        #         p_table_ind = p_table_ind + 1
            
        # Build shift decision array.
        # Recall that we decide for a shift (=1) if p-value <= sign level, otherwise not (=0).
        shift_decision_array[shift_decision_array <= sign_level] = 0
        shift_decision_array[shift_decision_array > sign_level] = 1
        shift_decision_array = 1 - shift_decision_array
        
        # ------------------- DR table
        
        # Sum detections over all shifts.
        shift_decision_array_dr = np.sum(shift_decision_array, 1)
        # Sum detections over all random runs.
        shift_decision_array_dr = np.sum(shift_decision_array_dr, 1)
        
        # Transpose from (len(samples), len(dr_techniques)) to (len(dr_techniques), len(samples)).
        shift_decision_array_dr = np.transpose(shift_decision_array_dr)
        
        # Some univariate shift experiments only have samples until 1000 due to the structure of the shift.
        if shift_decision_array_dr.shape[1] == 7 and test_type == 'univ':
            
            # Duplicate last column to create an 8th column with same results.
            additional_col = np.zeros((len(dr_techniques), 1))
            additional_col[:,0] = np.array(shift_decision_array_dr[:,-1])
            shift_decision_array_dr = np.append(shift_decision_array_dr, additional_col, axis=1)
            
            # Update DR table with new counts.
            dr_acc_table[:, :, 1] = dr_acc_table[:, :, 1] + shift_decision_array_dr
            dr_acc_table[:, :, 2] = dr_acc_table[:, :, 2] + np.ones((len(dr_techniques), len(samples) + 1)) * len(shifts) * rand_runs
            
        else:
            dr_acc_table[:, :, 1] = dr_acc_table[:, :, 1] + shift_decision_array_dr
            dr_acc_table[:, :, 2] = dr_acc_table[:, :, 2] + np.ones((len(dr_techniques), len(samples))) * len(shifts) * rand_runs
            
        # ------------------- Shift table
        
        # Sum detections over all random runs.
        shift_decision_array_sh = np.sum(shift_decision_array, 2)
        # Select best performing DR technique.
        if test_type == 'multiv':
            shift_decision_array_sh = shift_decision_array_sh[:,:,3]
        else:
            shift_decision_array_sh = shift_decision_array_sh[:,:,5]

        # Shift percentage
        if shift_decision_array_sh.shape[0] == 7 and test_type == 'univ':
            shift_decision_array_sh_0 = shift_decision_array_sh[:, 0]
            shift_decision_array_sh_0 = np.append(shift_decision_array_sh_0, [shift_decision_array_sh_0[-1]], axis=0)
            shift_decision_array_sh_1 = shift_decision_array_sh[:, 1]
            shift_decision_array_sh_1 = np.append(shift_decision_array_sh_1, [shift_decision_array_sh_1[-1]], axis=0)
            shift_decision_array_sh_2 = shift_decision_array_sh[:, 2]
            shift_decision_array_sh_2 = np.append(shift_decision_array_sh_2, [shift_decision_array_sh_2[-1]], axis=0)
            perc_acc_table[0, :, 1] = perc_acc_table[0, :, 1] + shift_decision_array_sh_0
            perc_acc_table[1, :, 1] = perc_acc_table[1, :, 1] + shift_decision_array_sh_1
            perc_acc_table[2, :, 1] = perc_acc_table[2, :, 1] + shift_decision_array_sh_2
            perc_acc_table[0, :, 2] = perc_acc_table[0, :, 2] + np.ones(len(samples)+1) * 1 * rand_runs * 1
            perc_acc_table[1, :, 2] = perc_acc_table[1, :, 2] + np.ones(len(samples)+1) * 1 * rand_runs * 1
            perc_acc_table[2, :, 2] = perc_acc_table[2, :, 2] + np.ones(len(samples)+1) * 1 * rand_runs * 1
        else:
            perc_acc_table[0, :, 1] = perc_acc_table[0, :, 1] + shift_decision_array_sh[:, 0]
            perc_acc_table[1, :, 1] = perc_acc_table[1, :, 1] + shift_decision_array_sh[:, 1]
            perc_acc_table[2, :, 1] = perc_acc_table[2, :, 1] + shift_decision_array_sh[:, 2]
            perc_acc_table[0, :, 2] = perc_acc_table[0, :, 2] + np.ones(len(samples)) * 1 * rand_runs * 1
            perc_acc_table[1, :, 2] = perc_acc_table[1, :, 2] + np.ones(len(samples)) * 1 * rand_runs * 1
            perc_acc_table[2, :, 2] = perc_acc_table[2, :, 2] + np.ones(len(samples)) * 1 * rand_runs * 1

        # Sum detections over all specific shifts in the shift class.
        shift_decision_array_sh = np.sum(shift_decision_array_sh, 1)
        
        # Same as above ...
        if shift_decision_array_sh.shape[0] == 7 and test_type == 'univ':
            shift_decision_array_sh = np.append(shift_decision_array_sh, [shift_decision_array_sh[-1]], axis=0)

            if shift_class in ['small_gn_shift', 'small_image_shift', 'ko_shift']:
                int_acc_table[0, :, 1] = int_acc_table[0, :, 1] + shift_decision_array_sh
                int_acc_table[0, :, 2] = int_acc_table[0, :, 2] + np.ones(len(samples) + 1) * 1 * rand_runs * len(shifts)
            elif shift_class in ['medium_gn_shift', 'medium_image_shift', 'adversarial_shift']:
                int_acc_table[1, :, 1] = int_acc_table[1, :, 1] + shift_decision_array_sh
                int_acc_table[1, :, 2] = int_acc_table[1, :, 2] + np.ones(len(samples) + 1) * 1 * rand_runs * len(
                    shifts)
            else:
                int_acc_table[2, :, 1] = int_acc_table[2, :, 1] + shift_decision_array_sh
                int_acc_table[2, :, 2] = int_acc_table[2, :, 2] + np.ones(len(samples) + 1) * 1 * rand_runs * len(
                    shifts)

            shift_acc_table[shift_class_idx, :, 1] = shift_acc_table[shift_class_idx, :, 1] + shift_decision_array_sh
            shift_acc_table[shift_class_idx, :, 2] = shift_acc_table[shift_class_idx, :, 2] + \
                                                     np.ones(len(samples)+1) * 1 * rand_runs * len(shifts)
        else:

            if shift_class in ['small_gn_shift', 'small_image_shift', 'ko_shift']:
                int_acc_table[0, :, 1] = int_acc_table[0, :, 1] + shift_decision_array_sh
                int_acc_table[0, :, 2] = int_acc_table[0, :, 2] + np.ones(len(samples)) * 1 * rand_runs * len(shifts)
            elif shift_class in ['medium_gn_shift', 'medium_image_shift', 'adversarial_shift']:
                int_acc_table[1, :, 1] = int_acc_table[1, :, 1] + shift_decision_array_sh
                int_acc_table[1, :, 2] = int_acc_table[1, :, 2] + np.ones(len(samples)) * 1 * rand_runs * len(
                    shifts)
            else:
                int_acc_table[2, :, 1] = int_acc_table[2, :, 1] + shift_decision_array_sh
                int_acc_table[2, :, 2] = int_acc_table[2, :, 2] + np.ones(len(samples)) * 1 * rand_runs * len(
                    shifts)

            shift_acc_table[shift_class_idx, :, 1] = shift_acc_table[shift_class_idx, :, 1] + shift_decision_array_sh
            shift_acc_table[shift_class_idx, :, 2] = shift_acc_table[shift_class_idx, :, 2] + \
                                                     np.ones(len(samples)) * 1 * rand_runs * len(shifts)
        
# Calculate accuracy values based on counts in other slices of the tensor.
dr_acc_table[:, :, 0] = np.divide(dr_acc_table[:, :, 1], dr_acc_table[:, :, 2])
shift_acc_table[:, :, 0] = np.divide(shift_acc_table[:, :, 1], shift_acc_table[:, :, 2])

int_acc_table[:, :, 0] = np.divide(int_acc_table[:, :, 1], int_acc_table[:, :, 2])
perc_acc_table[:, :, 0] = np.divide(perc_acc_table[:, :, 1], perc_acc_table[:, :, 2])

# Print results to console.
print('--------------------------')
print('DR table:')
print('--------------------------')
print(np.around(dr_acc_table[:, :, 0], decimals=2))
print('--------------------------')
print('Shift table:')
print('--------------------------')
print(np.around(shift_acc_table[:, :, 0], decimals=2))
print('--------------------------')
print('Intensity table:')
print('--------------------------')
print(np.around(int_acc_table[:, :, 0], decimals=2))
print('--------------------------')
print('Percentage table:')
print('--------------------------')
print(np.around(perc_acc_table[:, :, 0], decimals=2))

# Save results into results directory.
np.savetxt("%s/dr_table.csv" % path, dr_acc_table[:, :, 0], fmt='%.2f', delimiter=",")
np.savetxt("%s/shift_table.csv" % path, shift_acc_table[:, :, 0], fmt='%.2f', delimiter=",")
np.savetxt("%s/int_table.csv" % path, int_acc_table[:, :, 0], fmt='%.2f', delimiter=",")
np.savetxt("%s/per_table.csv" % path, perc_acc_table[:, :, 0], fmt='%.2f', delimiter=",")

dr_p_mean = np.mean(dr_p_table, axis=2)
dr_p_std = np.std(dr_p_table, axis=2)

# for dr_idx, dr in enumerate(dr_techniques):
#     errorfill(np.array(samples), dr_p_mean[dr_idx,:], dr_p_std[dr_idx,:], fmt=format[dr], color=colors[dr],
#               label="%s" % DimensionalityReduction(dr).name)
# plt.axhline(y=sign_level, color='k')
# plt.xlabel('Number of samples from test')
# plt.ylabel('$p$-value')
# plt.savefig("%s/shift-p-vals.pdf" % path, bbox_inches='tight')
# plt.legend()
# plt.savefig("%s/shift-p-vals-leg.pdf" % path, bbox_inches='tight')
# plt.clf()
