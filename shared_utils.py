# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from enum import Enum

# -------------------------------------------------
# ENUMS
# -------------------------------------------------

class TestDimensionality(Enum):
    One = 1
    Multi = 2
    Bin = 3


class OnedimensionalTest(Enum):
    KS = 1
    AD = 2


class MultidimensionalTest(Enum):
    MMD = 1
    KNN = 2
    FR = 3
    Energy = 4


class DimensionalityReduction(Enum):
    NoRed = 0
    PCA = 1
    SRP = 2
    UAE = 3
    TAE = 4
    BBSDs = 5
    BBSDh = 6
    Classif = 7
