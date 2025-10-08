import pytest
import radial_distr.find_inside_out_points as rif
import sys
import numpy as np
import os, os.path
import filecmp


def test_main_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "radial_distr.find_inside_out_points" in sys.modules

def test_main():
    '''Test the main function to be sure it processes the files correctly, and the generated file is compared to a reference file for correctness'''
    cwd = os.getcwd()
    print("Current directory", cwd)

    #Define file paths
    path = os.path.dirname(__file__)
    print ("stl file", path)
    stl_file_path = os.path.join(path,"data/ala.stl")
    dx_file_path  = os.path.join(path,"data/guv.O.5.dx")
    output_file_path = os.path.join(path,"data/inside_out_labels.dx")
    ref_file_path = os.path.join(path,"data/ref_ala_mask.dx")
    rif.main(stl_file_path, dx_file_path, output_file_path)

    #Compare reference file to generated file
    assert os.path.exists(output_file_path), "Generated file not found."
    assert filecmp.cmp(output_file_path, ref_file_path, shallow=False), "Generated file does not match reference file."

    assert True

    
