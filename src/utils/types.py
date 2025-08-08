import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from enum import Enum


class CellType(Enum):
    EMPTY = 0
    FLUID = 1
    OBSTACLE = 2

    EMWAVE = 4

    PERIODIC = 8
    DIRICHLET = 9
    NEUMANN_INFLOW = 10
    NEUMANN_OUTFLOW = 11
    FARFIELD = 12  # Far-field for fluid simulaion/ ABC for EM solver

    def __int__(self):
        return self.value

    @staticmethod
    def get_colormap():  # pragma: no cover
        """Compute a colormap to plot CellTypes with reasonable colors

        Returns:
            colormap, formatter, norm and norm bins
        """
        col_dict = {
            0: "white",
            1: "cyan",
            2: "black",
            4: "purple",
            8: "gray",
            9: "green",
            10: "blue",
            11: "orange",
            12: "navy",
        }
        # We create a colormap from our list of colors
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        labels = np.array(
            [
                "empty",
                "fluid",
                "obstacle",
                "EM wave",
                "B.C. Period",
                "B.C. Dirichlet",
                "B.C. Neumann Inflow",
                "B.C. Neumann Outflow",
                "B.C. Far field",
            ]
        )
        len_lab = len(labels)
        ## Prepare bins for the normalizer
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

        norm = BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = FuncFormatter(lambda x, pos: labels[norm(x)])
        return cm, fmt, norm, norm_bins


class VertexType(Enum):
    AIR = 0
    FLUID = 1
    OBSTACLE = 2

    AIR_FLUID = 3
    FLUID_OBSTACLE = 4
    OBSTACLE_AIR = 5

    TRIPLE_JUNCTION = 6

    def __int__(self):
        return self.value


class BoundaryType(Enum):
    DIRICHLET = 0
    NEUMANN = 1

    def __int__(self):
        return self.value


class LayerType(Enum):
    RHS = 0
    SINGLE_LAYER = 1
    DOUBLE_LAYER = 2
    HYPERSINGULAR_LAYER = 3

    def __int__(self):
        return self.value


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

    def __int__(self):
        return self.value


class BEMType(Enum):
    LAPLACE = 0
    HELMHOLTZ_BEM = 1
    HELMHOLTZ_MOM = 2

    def __int__(self):
        return self.value

    @staticmethod
    def use_complex(BEM_type: int) -> bool:
        if BEM_type == int(BEMType.LAPLACE):
            return False
        elif BEM_type == int(BEMType.HELMHOLTZ_BEM):
            return True
        elif BEM_type == int(BEMType.HELMHOLTZ_MOM):
            return True

        # else:
        return False

    @staticmethod
    def belong_Helmholtz(BEM_type: int) -> bool:
        if BEM_type == int(BEMType.HELMHOLTZ_BEM):
            return True
        elif BEM_type == int(BEMType.HELMHOLTZ_MOM):
            return True

        # else:
        return False


class OrderType(Enum):
    PLANAR = 0
    LINEAR = 1

    def __int__(self):
        return self.value


class PanelRelationType(Enum):
    SEPARATE = 0
    SAME_VERTEX = 1
    SAME_EDGE = 2
    SAME_FACE = 3

    def __int__(self):
        return self.value


class TrainType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    def __int__(self):
        return self.value


class GenUVMode(Enum):
    PLANAR = 0
    SPHERICAL_HARMONICS = 1
    BOX = 2

    def __int__(self):
        return self.value
