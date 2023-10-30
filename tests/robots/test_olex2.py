import os
import shutil

from iotbx import cif
import numpy as np
from qcrboxtools.robots.olex2 import Olex2Socket
#def test_olex2headless_available():
#    assert 'OLEX2HEADLESS' in os.environ, 'The OLEX2HEADLESS environment variable needs to be set to the Olex2 headless path'

def test_olex2server():
    olex2 = Olex2Socket()
    assert olex2.check_connection(), 'Server did not respond with ready to status check, are environment variables OLEX2SERVER and OLEX2PORT available and is the Olex2 socket server started?'

def test_olex2headless_refine(tmp_path):
    work_path = os.path.join(tmp_path, 'work.cif')
    shutil.copy('./tests/robots/cif_files/refine_nonconv.cif', work_path)

    olex2 = Olex2Socket()
    olex2.structure_path = work_path
    refine_answer = olex2.refine()

    cif_work = cif.reader(work_path).model()['epoxide']

    target_path = './tests/robots/cif_files/refine_conv.cif'
    cif_target = cif.reader(target_path).model()['epoxide']

    refined_path = tmp_path / 'epoxide.cif'
    cif_refined = cif.reader(target_path).model()['epoxide']

    print('bla')
    for ij in (11, 22, 33, 12, 13, 23):
        key = f'_atom_site_aniso_U_{ij}'
        refined_vals = np.array(
            [float(val.split('(')[0]) for val in cif_refined[key]]
        )
        target_vals = np.array(
            [float(val.split('(')[0]) for val in cif_target[key]]
        )
        assert max(abs(refined_vals - target_vals)) < 1e-4


