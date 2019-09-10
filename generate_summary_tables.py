# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np

from shared_utils import *

import sys

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
dr_table = np.zeros((len(dr_techniques), len(samples), 3))
shift_table = np.zeros((len(shift_classes), len(samples), 3))

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
            dr_table[:,:,1] = dr_table[:,:,1] + shift_decision_array_dr
            dr_table[:,:,2] = dr_table[:,:,2] + np.ones((len(dr_techniques), len(samples)+1)) * len(shifts) * rand_runs
            
        else:
            dr_table[:,:,1] = dr_table[:,:,1] + shift_decision_array_dr
            dr_table[:,:,2] = dr_table[:,:,2] + np.ones((len(dr_techniques), len(samples))) * len(shifts) * rand_runs
            
        # ------------------- Shift table
        
        # Sum detections over all random runs.
        shift_decision_array_sh = np.sum(shift_decision_array, 2)
        # Sum detections over all DR techniques.
        shift_decision_array_sh = np.sum(shift_decision_array_sh, 2)
        # Sum detections over all specific shifts in the shift class.
        shift_decision_array_sh = np.sum(shift_decision_array_sh, 1)
        
        # Same as above ...
        if shift_decision_array_sh.shape[0] == 7 and test_type == 'univ':
            shift_decision_array_sh = np.append(shift_decision_array_sh, [shift_decision_array_sh[-1]], axis=0)
            shift_table[shift_class_idx,:,1] = shift_table[shift_class_idx,:,1] + shift_decision_array_sh
            shift_table[shift_class_idx,:,2] = shift_table[shift_class_idx,:,2] + \
                                               np.ones(len(samples)+1) * len(dr_techniques) * rand_runs * len(shifts)
        else:
            shift_table[shift_class_idx,:,1] = shift_table[shift_class_idx,:,1] + shift_decision_array_sh
            shift_table[shift_class_idx,:,2] = shift_table[shift_class_idx,:,2] + \
                                               np.ones(len(samples)) * len(dr_techniques) * rand_runs * len(shifts)
        
# Calculate accuracy values based on counts in other slices of the tensor.
dr_table[:,:,0] = np.divide(dr_table[:,:,1], dr_table[:,:,2])
shift_table[:,:,0] = np.divide(shift_table[:,:,1], shift_table[:,:,2])

# Print results to console.
print('--------------------------')
print('DR table:')
print('--------------------------')
print(np.around(dr_table[:,:,0], decimals=2))
print('--------------------------')
print('Shift table:')
print('--------------------------')
print(np.around(shift_table[:,:,0], decimals=2))

# Save results into results directory.
np.savetxt("%sdr_table.csv" % path, dr_table[:,:,0], fmt='%.2f', delimiter=",")
np.savetxt("%sshift_table.csv" % path, shift_table[:,:,0], fmt='%.2f', delimiter=",")
