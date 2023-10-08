import re
import numpy as np
from iotbx import cif

test_dev = """   7  -4  -5 2038.13 563.062   3
   7  -5   0 13213.3 1168.96   3
   7  -5  -1 2415.45 563.194   3
"""

def format_floats(val):
    if val < 0:
        return f'{val: .8f}'[:8]
    else:
        return f' {val:.8f}'[:8]

def cifdata_str_or_index(model, dataset):
    if isinstance(dataset, int):
        keys = list(model.keys())
        dataset = keys[dataset]
    return model[dataset]

def cif2hkl4(cif_path, cif_dataset, hkl_path):
    with open(cif_path, 'r', encoding='UTF-8') as fo:
        cif_content = fo.read()

    # is shelx cif
    search_shelx = re.search(r'_shelx_hkl_file\n;(.*?);', cif_content, flags=re.DOTALL)
    if search_shelx is not None:
        hkl_content = search_shelx.group(1)
    else:
        cif_data = cifdata_str_or_index(
            cif.reader(input_string=cif_content).model(),
            cif_dataset
        )

        if '_diffrn_refln_scale_group_code' in cif_data:
            use_entries = [
                np.array(cif_data['_diffrn_refln_index_h'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_k'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_l'], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_net']],
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_u']],
                np.array(cif_data['_diffrn_refln_scale_group_code'], dtype=np.int64),
            ]
            line_format = '{:4d}{:4d}{:4d}{}{}{:4d}'
        else:
            use_entries = [
                np.array(cif_data['_diffrn_refln_index_h'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_k'], dtype=np.int64),
                np.array(cif_data['_diffrn_refln_index_l'], dtype=np.int64),
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_net']],
                [format_floats(float(val)) for val in cif_data['_diffrn_refln_intensity_u']]
            ]
            line_format = '{:4d}{:4d}{:4d}{}{}'
        hkl_content = '\n'.join(line_format.format(*entryset) for entryset in zip(*use_entries))
    # is cif using the entries
    #print(hkl_content)
    with open(hkl_path, 'w', encoding='ASCII') as fo:
        fo.write(hkl_content)
