import re
import iotbx

test_dev = """   7  -4  -5 2038.13 563.062   3
   7  -5   0 13213.3 1168.96   3
   7  -5  -1 2415.45 563.194   3
"""

def cif2hkl(cif_path, hkl_path, hkl_format):
    with open(cif_path, 'r', encoding='UTF-8') as fo:
        cif_content = fo.read()

    # is shelx cif
    search_shelx = re.search(r'_shelx_hkl_file\n;(.*?);', cif_content, flags=re.DOTALL)
    if search_shelx is not None:
        hkl_content = search_shelx.group(1)
    else:
        hkl_content = test_dev

    # is cif using the entries
    with open(hkl_path, 'w', encoding='ASCII') as fo:
        fo.write(hkl_content)

def hkl2cif(hkl_name, hkl_format, cif_name):
    pass

