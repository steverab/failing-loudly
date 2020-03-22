# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
import torch
import random
from torch import *
from torch_two_sample import *
from scipy.stats import ks_2samp, binom_test, chisquare, chi2_contingency, anderson_ksamp
from scipy.spatial import distance

from shared_utils import *

# -------------------------------------------------
# SHIFT TESTER
# -------------------------------------------------

class ShiftTester:

    def __init__(self, dim=TestDimensionality.One, sign_level=0.05, ot=None, mt=None):
        self.dim = dim
        self.sign_level = sign_level
        self.ot = ot
        self.mt = mt

    def test_shift(self, X_tr, X_te):
        if self.ot is not None:
            return self.one_dimensional_test(X_tr, X_te)
        elif self.mt is not None:
            return self.multi_dimensional_test(X_tr, X_te)

    def test_chi2_shift(self, X_tr, X_te, nb_classes):

        # Calculate observed and expected counts
        freq_exp = np.zeros(nb_classes)
        freq_obs = np.zeros(nb_classes)

        unique_tr, counts_tr = np.unique(X_tr, return_counts=True)
        total_counts_tr = np.sum(counts_tr)
        unique_te, counts_te = np.unique(X_te, return_counts=True)
        total_counts_te = np.sum(counts_te)

        for i in range(len(unique_tr)):
            val = counts_tr[i]
            freq_exp[unique_tr[i]] = val
            
        for i in range(len(unique_te)):
            freq_obs[unique_te[i]] = counts_te[i]

        if np.amin(freq_exp) == 0 or np.amin(freq_obs) == 0:
            # The chi-squared test using contingency tables is not well defined if zero-element classes exist, which
            # might happen in the low-sample regime. In this case, we calculate the standard chi-squared test.
            #for i in range(len(unique_tr)):
            #    val = counts_tr[i] / total_counts_tr * total_counts_te
            #    freq_exp[unique_tr[i]] = val
            #_, p_val = chisquare(freq_obs, f_exp=freq_exp)
            p_val = random.uniform(0, 1)
        else:
            # In almost all cases, we resort to obtaining a p-value from the chi-squared test's contingency table.
            freq_conc = np.array([freq_exp, freq_obs])
            _, p_val, _, _ = chi2_contingency(freq_conc)
        
        return p_val

    def test_shift_bin(self, k, n, test_rate):
        p_val = binom_test(k, n, test_rate)
        return p_val

    def one_dimensional_test(self, X_tr, X_te):
        p_vals = []

        # For each dimension we conduct a separate KS test
        for i in range(X_tr.shape[1]):
            feature_tr = X_tr[:, i]
            feature_te = X_te[:, i]

            t_val, p_val = None, None

            if self.ot == OnedimensionalTest.KS:

                # Compute KS statistic and p-value
                t_val, p_val = ks_2samp(feature_tr, feature_te)
            elif self.ot == OnedimensionalTest.AD:
                t_val, _, p_val = anderson_ksamp([feature_tr.tolist(), feature_te.tolist()])

            p_vals.append(p_val)

        # Apply the Bonferroni correction to bound the family-wise error rate. This can be done by picking the minimum
        # p-value from all individual tests.
        p_vals = np.array(p_vals)
        p_val = min(np.min(p_vals), 1.0)

        return p_val, p_vals

    def multi_dimensional_test(self, X_tr, X_te):

        # torch_two_sample somehow wants the inputs to be explicitly casted to float 32.
        X_tr = X_tr.astype(np.float32)
        X_te = X_te.astype(np.float32)

        p_val = None

        # We provide a couple of different tests, although we only report results for MMD in the paper.
        if self.mt == MultidimensionalTest.MMD:
            mmd_test = MMDStatistic(len(X_tr), len(X_te))

            # As per the original MMD paper, the median distance between all points in the aggregate sample from both
            # distributions is a good heuristic for the kernel bandwidth, which is why compute this distance here.
            if len(X_tr.shape) == 1:
                X_tr = X_tr.reshape((len(X_tr),1))
                X_te = X_te.reshape((len(X_te),1))
                all_dist = distance.cdist(X_tr, X_te, 'euclidean')
            else:
                all_dist = distance.cdist(X_tr, X_te, 'euclidean')
            median_dist = np.median(all_dist)

            # Calculate MMD.
            t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)),
                                     alphas=[1/median_dist], ret_matrix=True)
            p_val = mmd_test.pval(matrix)
        elif self.mt == MultidimensionalTest.Energy:
            energy_test = EnergyStatistic(len(X_tr), len(X_te))
            t_val, matrix = energy_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                        torch.autograd.Variable(torch.tensor(X_te)),
                                        ret_matrix=True)
            p_val = energy_test.pval(matrix)
        elif self.mt == MultidimensionalTest.FR:
            fr_test = FRStatistic(len(X_tr), len(X_te))
            t_val, matrix = fr_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                    torch.autograd.Variable(torch.tensor(X_te)),
                                    norm=2, ret_matrix=True)
            p_val = fr_test.pval(matrix)
        elif self.mt == MultidimensionalTest.KNN:
            knn_test = KNNStatistic(len(X_tr), len(X_te), 20)
            t_val, matrix = knn_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)),
                                     norm=2, ret_matrix=True)
            p_val = knn_test.pval(matrix)
            
        return p_val, np.array([])
