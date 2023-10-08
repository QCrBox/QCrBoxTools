import re
import numpy as np
from iotbx import cif

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

    #TODO support mixed Olex/SHELX cifs
    with open(cif_path, 'r', encoding='UTF-8') as fo:
        cif_content = fo.read()

    # is shelx cif
    search_shelx = re.findall(r'_shelx_hkl_file\n;(.*?);', cif_content, flags=re.DOTALL)
    if len(search_shelx) > 0:
        if isinstance(cif_dataset, int):
            hkl_content = search_shelx[cif_dataset]
        else:
            data_strings = re.findall(r'data_(.*?)\n', cif_content)
            hkl_content = search_shelx[data_strings.index(cif_dataset)]
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
