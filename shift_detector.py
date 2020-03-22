# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from shift_tester import *
from shift_reductor import *
from shared_utils import *

# -------------------------------------------------
# SHIFT DETECTOR
# -------------------------------------------------


class ShiftDetector:

    # Model storage for quick access.
    red_models = [None] * len(DimensionalityReduction)

    def __init__(self, dr_techniques, test_types, od_tests, md_tests, sign_level, red_models, sample, datset):
        self.dr_techniques = dr_techniques
        self.test_types = test_types
        self.od_tests = od_tests
        self.md_tests = md_tests
        self.sign_level = sign_level
        self.red_models = red_models
        self.sample = sample
        self.datset = datset
        
    def classify_data(self, X_tr, y_tr, X_val, y_val, X_te, y_te, orig_dims, nb_classes):
        shift_reductor = ShiftReductor(X_tr, y_tr, X_val, y_val, DimensionalityReduction.BBSDh, orig_dims, self.datset, dr_amount=32)
        shift_reductor_model = shift_reductor.fit_reductor()
        X_te_red = shift_reductor.reduce(shift_reductor_model, X_te)
        return X_te_red

    def detect_data_shift(self, X_tr, y_tr, X_val, y_val, X_te, y_te, orig_dims, nb_classes):
        od_decs = np.ones(len(self.dr_techniques)) * (-1)
        ind_od_decs = np.ones((len(self.dr_techniques), len(self.od_tests))) * (-1)
        ind_od_p_vals = np.ones((len(self.dr_techniques), len(self.od_tests))) * (-1)

        md_decs = np.ones(len(self.dr_techniques)) * (-1)
        ind_md_decs = np.ones((len(self.dr_techniques), len(self.md_tests))) * (-1)
        ind_md_p_vals = np.ones((len(self.dr_techniques), len(self.md_tests))) * (-1)

        red_dim = -1
        
        val_acc = None
        te_acc = None

        # For all dimensionality reduction techniques:
        # 1. Train/Load model.
        # 2. Reduce inputs to latent representations.
        # 3. Perform shift detection test.
        for dr_ind, dr_technique in enumerate(self.dr_techniques):
            
            print(DimensionalityReduction(dr_technique).name)

            # Train or load reduction model.
            shift_reductor = ShiftReductor(X_tr, y_tr, X_val, y_val, DimensionalityReduction(dr_technique), orig_dims, self.datset, dr_amount=32)
            red_dim = shift_reductor.dr_amount
            shift_reductor_model = None
            if self.red_models[dr_ind] is None:
                shift_reductor_model = shift_reductor.fit_reductor()
                self.red_models[dr_ind] = shift_reductor_model
            else:
                shift_reductor_model = self.red_models[dr_ind]

            # Reduce validation and test set.
            X_tr_red = shift_reductor.reduce(shift_reductor_model, X_val)
            X_te_red = shift_reductor.reduce(shift_reductor_model, X_te)

            # Compute classification accuracy on both sets for malignancy detection.
            if dr_technique == DimensionalityReduction.BBSDh.value:
                val_acc = np.sum(np.equal(X_tr_red, y_val).astype(int))/X_tr_red.shape[0]
                te_acc = np.sum(np.equal(X_te_red, y_te).astype(int))/X_te_red.shape[0]

            od_loc_p_vals = []
            md_loc_p_vals = []

            # Iterate over all test types and use appropriate test for the DR technique used.
            for test_type in self.test_types:
                if test_type == TestDimensionality.One.value:
                    for od_test in self.od_tests:
                        shift_tester = ShiftTester(TestDimensionality(test_type), sign_level=self.sign_level, ot=OnedimensionalTest(od_test))
                        if dr_technique != DimensionalityReduction.BBSDh.value:
                            p_val, feature_p_vals = shift_tester.test_shift(X_tr_red, X_te_red)
                        else:
                            p_val = shift_tester.test_chi2_shift(X_tr_red, X_te_red, nb_classes)
                        od_loc_p_vals.append(p_val)
                if test_type == TestDimensionality.Multi.value:
                    for md_test in self.md_tests:
                        shift_tester = ShiftTester(TestDimensionality(test_type), sign_level=self.sign_level, mt=MultidimensionalTest(md_test))
                        p_val, _ = shift_tester.test_shift(X_tr_red[:self.sample], X_te_red)
                        md_loc_p_vals.append(p_val)

            if dr_technique != DimensionalityReduction.BBSDh.value:
                # Lower the significance level for all tests (Bonferroni) besides BBSDh, which needs no correction.
                adjust_sign_level = self.sign_level / X_tr_red.shape[1]
            else: 
                adjust_sign_level = self.sign_level

            # Compute shift decisions (not currently used, we resort to generate_summary_tables.py)
            od_loc_decs = np.array([1 if val < adjust_sign_level else 0 for val in od_loc_p_vals])
            md_loc_decs = np.array([1 if val < self.sign_level else 0 for val in md_loc_p_vals])

            # Rescale p-values to "normal" scale, since many p-values correspond to the minimum. This makes it easier
            # later to plot them all on the same scale with the standard significance level.
            od_loc_p_vals = np.array(od_loc_p_vals)
            if dr_technique != DimensionalityReduction.BBSDh.value:
                od_loc_p_vals = od_loc_p_vals * X_tr_red.shape[1]
                od_loc_p_vals[od_loc_p_vals > 1] = 1

            # Identify best method across tests (not currently used)
            if len(od_loc_decs) > 0:
                od_decs[dr_ind] = np.max(od_loc_decs)
                ind_od_decs[dr_ind, :] = od_loc_decs
                ind_od_p_vals[dr_ind, :] = od_loc_p_vals

            # Same as above ...
            if len(md_loc_decs) > 0:
                md_decs[dr_ind] = np.max(md_loc_decs)
                ind_md_decs[dr_ind, :] = md_loc_decs
                ind_md_p_vals[dr_ind, :] = np.array(md_loc_p_vals)

        return (od_decs, ind_od_decs, ind_od_p_vals), (md_decs, ind_md_decs, ind_md_p_vals), red_dim, self.red_models, val_acc, te_acc

