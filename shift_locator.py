# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from shift_tester import *

import keras
import keras_resnet
from keras import optimizers


class DifferenceClassifier(Enum):
    FFNNDCL = 1
    FLDA = 2


class AnomalyDetection(Enum):
    OCSVM = 1

# -------------------------------------------------
# SHIFT LOCATOR
# -------------------------------------------------


class ShiftLocator:

    # Shuffle two sets in unison, specifically used for data points and labels.
    def __unison_shuffled_copies(self, a, b, c):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]

    # Partition the data set(s) for the difference classifier.
    def __prepare_difference_detector(self, x_train, y_train, x_test, y_test, balanced):

        # Balancing makes testing easier.
        if balanced:
            if len(x_train) > len(x_test):
                x_train = x_train[:len(x_test)]
                y_train = y_train[:len(y_test)]
            else:
                x_test = x_test[:len(x_train)]
                y_test = y_test[:len(y_test)]

        # Extract halves from both sets
        x_train_first_half = x_train[:len(x_train) // 2, :]
        y_train_first_half = y_train[:len(y_train) // 2]
        x_train_second_half = x_train[len(x_train) // 2:, :]
        y_train_second_half = y_train[len(y_train) // 2:]
        x_test_first_half = x_test[:len(x_test) // 2, :]
        y_test_first_half = y_test[:len(y_test) // 2]
        x_test_second_half = x_test[len(x_test) // 2:, :]
        y_test_second_half = y_test[len(y_test) // 2:]

        self.ratio = len(x_train_first_half) / (len(x_train_first_half) + len(x_test_first_half))

        # Recombine halves into new dataset, where samples from source are labeled with 0 and target samples with 1.
        x_train_new = np.append(x_train_first_half, x_test_first_half, axis=0)
        y_train_old = np.append(y_train_first_half, y_test_first_half)
        y_train_new = np.zeros(len(x_train_new))
        y_train_new[len(x_train_first_half):] = np.ones(len(x_test_first_half))
        x_test_new = np.append(x_train_second_half, x_test_second_half, axis=0)
        y_test_old = np.append(y_train_second_half, y_test_second_half)
        y_test_new = np.zeros(len(x_test_new))
        y_test_new[len(x_train_second_half):] = np.ones(len(x_test_second_half))

        x_train_new, y_train_new, y_train_old = self.__unison_shuffled_copies(x_train_new, y_train_new, y_train_old)
        x_test_new, y_test_new, y_test_old = self.__unison_shuffled_copies(x_test_new, y_test_new, y_test_old)

        return x_train_new, y_train_new, y_train_old, x_test_new, y_test_new, y_test_old

    def __init__(self, orig_dims, dc=None, ad=None, sign_level=0.05):
        self.orig_dims = orig_dims
        self.dc = dc
        self.ad = ad
        self.sign_level = sign_level
        self.ratio = -1.0

    def build_model(self, X_tr, y_tr, X_te, y_te, balanced=True):
        if self.dc == DifferenceClassifier.FFNNDCL:
            return self.neural_network_difference_detector(X_tr, y_tr, X_te, y_te, bal=balanced)
        elif self.dc == DifferenceClassifier.FLDA:
            return self.fisher_lda_difference_detector(X_tr, X_te, balanced=balanced)
        elif self.ac == AnomalyDetection.OCSVM:
            return self.one_class_svm(X_tr, X_te, balanced=balanced)

    def most_likely_shifted_samples(self, model, X_te_new, y_te_new):
        if self.dc == DifferenceClassifier.FFNNDCL:

            # Predict class assignments.
            y_te_new_pred = model.predict(X_te_new.reshape(len(X_te_new), self.orig_dims[0], self.orig_dims[1],
                                                           self.orig_dims[2]))

            # Get most anomalous indices sorted in descending order.
            most_conf_test_indices = np.argsort(y_te_new_pred[:,1])[::-1]
            most_conf_test_perc = np.sort(y_te_new_pred[:,1])[::-1]

            # Test whether classification accuracy is statistically significant.
            y_te_new_pred_argm = np.argmax(y_te_new_pred, axis=1)
            errors = np.count_nonzero(y_te_new - y_te_new_pred_argm)
            successes = len(y_te_new_pred_argm) - errors
            shift_tester = ShiftTester(TestDimensionality.Bin)
            p_val = shift_tester.test_shift_bin(successes, len(y_te_new_pred_argm), self.ratio)

            return most_conf_test_indices, most_conf_test_perc, p_val < self.sign_level, p_val
        if self.dc == DifferenceClassifier.FLDA:
            y_te_new_pred = model.predict(X_te_new)

            y_te_new_pred_probs = model.predict_proba(X_te_new)
            most_conf_test_indices = np.argsort(y_te_new_pred_probs[:, 1])[::-1]
            most_conf_test_perc = np.sort(y_te_new_pred_probs[:, 1])[::-1]

            novelties = X_te_new[y_te_new_pred == 1]
            errors = np.count_nonzero(y_te_new - y_te_new_pred)
            successes = len(y_te_new_pred) - errors
            shift_tester = ShiftTester(TestDimensionality.Bin)
            p_val = shift_tester.test_shift_bin(successes, len(y_te_new_pred), self.ratio)
            return most_conf_test_indices, most_conf_test_perc, p_val < self.sign_level, p_val
        elif self.ad == AnomalyDetection.OCSVM:
            y_pred_te = model.predict(X_te_new)
            novelties = X_te_new[y_pred_te == -1]
            return novelties, None, -1

    def fisher_lda_difference_detector(self, X_tr, X_te, balanced=False):
        X_tr_dcl, y_tr_dcl, X_te_dcl, y_te_dcl = self.__prepare_difference_detector(X_tr, X_te, balanced=balanced)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_tr_dcl, y_tr_dcl)
        return lda, None, (X_tr_dcl, y_tr_dcl, X_te_dcl, y_te_dcl)

    def neural_network_difference_detector(self, X_tr, y_tr, X_te, y_te, bal=False):
        D = X_tr.shape[1]
        X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old = self.__prepare_difference_detector(X_tr, y_tr,
                                                                                                        X_te, y_te,
                                                                                                        balanced=bal)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        early_stopper = EarlyStopping(min_delta=0.001, patience=10)
        batch_size = 128
        nb_classes = 2
        epochs = 200
        
        model = keras_resnet.models.ResNet18(keras.layers.Input(self.orig_dims), classes=nb_classes)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9),
                      metrics=['accuracy'])
        
        model.fit(X_tr_dcl.reshape(len(X_tr_dcl), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                  to_categorical(y_tr_dcl),
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_te_dcl.reshape(len(X_te_dcl), self.orig_dims[0], self.orig_dims[1],
                                                    self.orig_dims[2]), to_categorical(y_te_dcl)),
                  shuffle=True,
                  callbacks=[lr_reducer, early_stopper])

        score = model.evaluate(X_te_dcl.reshape(len(X_te_dcl), self.orig_dims[0], self.orig_dims[1],
                                                self.orig_dims[2]), to_categorical(y_te_dcl))

        return model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old)

    def one_class_svm(self, X_tr, X_te, balanced=False):
        X_tr_dcl, y_tr_dcl, X_te_dcl, y_te_dcl = self.__prepare_difference_detector(X_tr, X_te, balanced=balanced)
        svm = OneClassSVM()
        svm.fit(X_tr_dcl)
        return svm, None, (X_tr_dcl, y_tr_dcl, X_te_dcl, y_te_dcl)
