import pytest
import radial_distr.read_in_files as rif
import sys
import numpy as np
import os
import filecmp


def test_main_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "radial_distr.read_in_files" in sys.modules

def test_main():
    '''Test the main function to be sure it processes the files correctly, and the generated file is compared to a reference file for correctness'''
    cwd = os.getcwd()
    print("Current directory", cwd)

    #Define file paths
    stl_file_path = "tests/data/ala.stl"
    dx_file_path  = "tests/data/guv.O.5.dx"
    ref_file_path = "tests/data/ref_ala_mask.dx"
    rif.main(stl_file_path, dx_file_path)
    generated_file_path = "tests/data/inside_out_labels.dx"

    #Call main function in read_in_files
    rif.main(stl_file_path, dx_file_path)

    #Compare reference file to generated file
    assert os.path.exists(generated_file_path), "Generated file not found."
    assert filecmp.cmp(generated_file_path, ref_file_path, shallow=False), "Generated file does not match reference file."

    #Should I check grid data such as shape?

    assert True