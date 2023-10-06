import pytest
import numpy as np

def test_cif_2_shelx_hkl(cif_name):

    # read shelxl hkl (created by olex)
    # convert into numpy arrays hkl, intensity, esd
    # sort arrays by h, k, l
    # create converted hkl from cif into temporary file
    # read file the same way
    # compare whether identical
    pass