"""
This test module provides functionalities to test the interactions of `Olex2Socket` with a running
Olex2 server. It contains tests to check the Olex2 server's availability and functionality of
headless refinement using the socket.
"""

import os
import shutil

from iotbx import cif
import numpy as np
from qcrboxtools.robots.olex2 import Olex2Socket

def test_olex2server():
    """
    Tests the connection to the Olex2 server using the `Olex2Socket` class. The test checks whether
    the server is ready and responsive. It expects Olex2 socket server to be started and if using
    a non-default address or port that the OLEX2SERVER and OLEX2PORT environment variables are set.
    """
    olex2 = Olex2Socket()
    message = (
        'Server did not respond with ready to status check, are environment variables OLEX2SERVER '
        + 'and OLEX2PORT available and is the Olex2 socket server started?'
    )
    assert olex2.check_connection(), message

def test_olex2_refine(tmp_path):
    """
    Tests the refinement functionality of `Olex2Socket`. The function simulates a
    refinement process by copying a non-converged CIF file to a temporary working path, and
    then triggering refinement using the `Olex2Socket` class. The test ensures that the refined
    values match the target values within a specified tolerance.

    Args:
    - tmp_path: A fixture provided by pytest for temporary directories.
    """
    work_path = os.path.join(tmp_path, 'work.cif')
    shutil.copy('./tests/robots/cif_files/refine_nonconv_nonHaniso.cif', work_path)

    olex2 = Olex2Socket()
    olex2.structure_path = work_path
    _ = olex2.refine()

    target_path = './tests/robots/cif_files/refine_conv_nonHaniso.cif'
    cif_target = cif.reader(str(target_path)).model()['epoxide']

    cif_refined = cif.reader(str(work_path)).model()['work']

    for ij in (11, 22, 33, 12, 13, 23):
        key = f'_atom_site_aniso_U_{ij}'
        refined_vals = np.array(
            [float(val.split('(')[0]) for val in cif_refined[key]]
        )
        target_vals = np.array(
            [float(val.split('(')[0]) for val in cif_target[key]]
        )
        assert max(abs(refined_vals - target_vals)) < 1.1e-4

def test_olex2_refine_tsc(tmp_path):
    work_path = os.path.join(tmp_path, 'work.cif')
    shutil.copy('./tests/robots/cif_files/refine_nonconv_allaniso.cif', work_path)

    tsc_path = os.path.join(tmp_path, 'work.tscb')
    shutil.copy('./tests/robots/cif_files/refine_allaniso.tscb', tsc_path)

    olex2 = Olex2Socket()
    olex2.structure_path = work_path

    olex2.tsc_path = tsc_path
    _ = olex2.refine()

    target_path = './tests/robots/cif_files/refine_conv_allaniso.cif'
    cif_target = cif.reader(str(target_path)).model()['epoxide']

    cif_refined = cif.reader(str(work_path)).model()['work']

    for ij in (11, 22, 33, 12, 13, 23):
        key = f'_atom_site_aniso_U_{ij}'
        refined_vals = np.array(
            [float(val.split('(')[0]) for val in cif_refined[key]]
        )
        target_vals = np.array(
            [float(val.split('(')[0]) for val in cif_target[key]]
        )
        assert max(abs(refined_vals - target_vals)) < 1.1e-4
