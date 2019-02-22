import numpy as np
import torch
from torch import *
from torch_two_sample import *
from scipy.stats import ks_2samp, binom_test, chisquare, chi2_contingency

from enum import Enum
from shared_utils import *


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

    def test_chi2_shift(self, X_tr, X_te):
        freq_exp = np.zeros(10)
        freq_obs = np.zeros(10)
        
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
            for i in range(len(unique_tr)):
                val = counts_tr[i] / total_counts_tr * total_counts_te
                freq_exp[unique_tr[i]] = val
            _, p_val = chisquare(freq_obs, f_exp=freq_exp)
        else:
            freq_conc = np.array([freq_exp, freq_obs])
            _, p_val, _, _ = chi2_contingency(freq_conc)
        
        return p_val

    def test_shift_bin(self, k, n, test_rate):
        p_val = binom_test(k, n, test_rate)
        return p_val

    def one_dimensional_test(self, X_tr, X_te):
        p_vals = []

        for i in range(X_tr.shape[1]):
            feature_tr = X_tr[:, i]
            feature_te = X_te[:, i]

            t_val, p_val = None, None

            if self.ot == OnedimensionalTest.KS:
                t_val, p_val = ks_2samp(feature_tr, feature_te)

            p_vals.append(p_val)

        p_vals = np.array(p_vals)
        p_val = min(np.min(p_vals), 1.0)

        return p_val, p_vals

    def multi_dimensional_test(self, X_tr, X_te):
        X_tr = X_tr.astype(np.float32)
        X_te = X_te.astype(np.float32)

        p_val = None

        if self.mt == MultidimensionalTest.MMD:
            mmd_test = MMDStatistic(len(X_tr), len(X_te))
            t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)), alphas=[1], ret_matrix=True)
            p_val = mmd_test.pval(matrix)
        elif self.mt == MultidimensionalTest.Energy:
            energy_test = EnergyStatistic(len(X_tr), len(X_te))
            t_val, matrix = energy_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                        torch.autograd.Variable(torch.tensor(X_te)), ret_matrix=True)
            p_val = energy_test.pval(matrix)
        elif self.mt == MultidimensionalTest.FR:
            fr_test = FRStatistic(len(X_tr), len(X_te))
            t_val, matrix = fr_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                    torch.autograd.Variable(torch.tensor(X_te)), norm=2, ret_matrix=True)
            p_val = fr_test.pval(matrix)
        elif self.mt == MultidimensionalTest.KNN:
            knn_test = KNNStatistic(len(X_tr), len(X_te), 20)
            t_val, matrix = knn_test(torch.autograd.Variable(torch.tensor(X_tr)),
                                     torch.autograd.Variable(torch.tensor(X_te)), norm=2, ret_matrix=True)
            p_val = knn_test.pval(matrix)
            
        return p_val, np.array([])
