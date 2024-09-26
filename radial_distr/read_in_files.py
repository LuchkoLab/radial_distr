import numpy as np
from gridData import Grid
from stl import mesh
import MDAnalysis as mda
import parmed as pmd
import mrcfile 

'''This module takes in .stl and .mrc or .dx data, and it determines whether points lie inside or outside of a surface created by a .stl file '''


def load_stl 
    #Read in .stl file
    ala_stl = mesh.Mesh.from_file('/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.stl')

    #test if file was loaded with print statement, we will try accessing the vertices of the mesh
    print(ala_stl.vectors.shape)

    #Access the number of facets
    print(f"Number of facets: {ala_stl.vectors.shape[0]}")

    #Access and print an array containing the normal vectors for each facet
    print(ala_stl.normals)

    #Print shape
    print(ala_stl.normals.shape)
def load_mrc  
    #Read in dx file
    grid = Grid('/home/tyork/ribosome/radial_distr/radial_distr/tests/guv.O.5.dx')

    Access the grid data
    data = grid.grid

    #test if data was loaded by getting grid shape and origin
    grid_shape = grid.grid.shape
    origin = grid.origin
        midpoints = grid.midpoints
        print("Grid Shape:", grid_shape)
    print("Origin:", origin)
    print("Midpoints:", midpoints)

