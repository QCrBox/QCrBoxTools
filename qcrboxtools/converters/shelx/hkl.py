import re
import iotbx

test_dev = """   7  -4  -5 2038.13 563.062   3
   7  -5   0 13213.3 1168.96   3
   7  -5  -1 2415.45 563.194   3
"""

def cif2hkl(cif_path, hkl_path, hkl_format):

    # is shelx cif
    with open(hkl_path, 'w', encoding='ASCII') as fo:
        fo.write(test_dev)

def hkl2cif(hkl_name, hkl_format, cif_name):
    pass

