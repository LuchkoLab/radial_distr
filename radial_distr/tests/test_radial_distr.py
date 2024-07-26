"""
Unit and regression test for the radial_distr package.
"""

# Import package, test suite, and other packages as needed
import sys

import pandas as pd

import os

import pytest

import subprocess

import radial_distr  

import gridData

import parmed as pmd 

import numpy as np

@pytest.fixture(scope="module")
def setup_files():
    # Get the absolute path of the directory containing this script
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # Paths to the input files (relative to the current script directory)
    parm7_file = os.path.join(current_dir, "ala.parm7")
    rst7_file = os.path.join(current_dir, "ala.rst7")
    guv_file = os.path.join(current_dir, "guv.O.5.dx")
    reference_output = os.path.join(current_dir, "ala-rdf.csv.ref")

    # Ensure the input files and reference output file exist
    assert os.path.isfile(parm7_file), f"Missing file: {parm7_file}"
    assert os.path.isfile(rst7_file), f"Missing file: {rst7_file}"
    assert os.path.isfile(guv_file), f"Missing file: {guv_file}"
    assert os.path.isfile(reference_output), f"Missing file: {reference_output}"

    # Yield the files for use in the test
    yield parm7_file, rst7_file, guv_file, reference_output

    # Clean up: Remove the newly generated output file if needed
    if os.path.isfile("rdf.csv"):
        os.remove("rdf.csv")

def test_radial_distr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "radial_distr" in sys.modules

def test_radial_distr(setup_files):
    parm7_file, rst7_file, guv_file, reference_output = setup_files

    g = gridData.Grid(guv_file)

    parm = pmd.load_file(parm7_file, rst7_file)

    rdf,dr = radial_distr.compute_rad_dist(parm.coordinates, g.grid, g.midpoints, max_cutoff=8)   

    df = pd.DataFrame({'sep': np.linspace(dr / 2, len(rdf) * dr - dr / 2, len(rdf)), 'rdf': rdf})
    df = df.fillna(0)  # Fill any NaN values with 0
  
    # Load the reference output
    
    reference_df = pd.read_csv(reference_output)
    
    #df_python = radial_distr.compute_rad_dist_python(parm.coordinates, g)
    #print(df_python)
    # Check if the dataframes are equal

    pd.testing.assert_frame_equal(df, reference_df)
       