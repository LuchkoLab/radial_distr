"""Provide the primary functions."""
import argparse
import numpy as np
import parmed as pmd
import pandas as pd
import gridData
import numba
import MDAnalysis
import multiprocessing

def main():
    """
    Main function to execute the radial distribution function computation.
    
    This function reads grid and molecular data files, computes the radial
    distribution function using Numba-optimized and non-optimized methods, and
    saves the results to CSV files.
    """
    args = get_args()  # Parse command-line arguments
    
    # Load the grid data from the specified file
    g = gridData.Grid(args.guv)
    print("grid shape: ", g.grid.shape)  # Print the shape of the grid
    print("number of dimensions: ", g.grid.ndim)  # Print the number of dimensions
    print("origin: ", g.origin)  # Print the origin of the grid

    # Load the molecular data from the specified files
    parm = pmd.load_file(args.parm, args.coords)
    print("test print", parm.coordinates)  # Print the molecular coordinates
    print(parm.coordinates.shape)  # Print the shape of the coordinates array
    rdf,dr = compute_rad_dist(parm.coordinates, g.grid, g.midpoints, args.cutoff, args.nprocess)
    # Create a DataFrame for the results and save it to a CSV file
    df = pd.DataFrame({'sep': np.linspace(dr / 2, len(rdf) * dr - dr / 2, len(rdf)), 'rdf': rdf})
    df = df.fillna(0)  # Fill any NaN values with 0
    # df = compute_rad_dist_python(parm.coordinates, g)
    # print(df)
    df.to_csv(args.rdf, index=False)

    # Compute the radial distribution function without Numba optimization
    rdf = compute_rad_dist(parm.coordinates, g)
    print(rdf)
    rdf.to_csv('rdf.csv', index=False)  # Save the DataFrame to a CSV file

def get_args():
    """
    Parse command-line arguments.

    Returns:
    argparse.Namespace: Parsed command-line arguments containing the paths to the guv file,
    parmtop file, and coordinate file.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(prog='radial_distr.py', description='This program creates a radial distribution function around a biological molecule')
    # Define expected command-line arguments
    parser.add_argument('--guv', help='guv file expected', required=True)
    parser.add_argument('-p', '--parm',help='parmtop file',required=True)
    parser.add_argument('-y', '--coords', help='coordinate file',required=True)
    parser.add_argument('--cutoff', type=float, default = 20., help = 'Maximum cutoff in Angstroms. This will be automatically reduced if greater than any of the grid dimensions.')
    parser.add_argument('--nprocess', type=int, default = 1, help = 'Number of parallel processes.')
    parser.add_argument('--rdf', required = True, help = 'Output rdf file')
    # Parse and return the argumets
    args = parser.parse_args()
    return args

def compute_rad_dist(coordinates, grid, midpoints, max_cutoff=20, nworkers = 1):
    base_tasks = grid.shape[0] // nworkers
    """
    Compute the radial distribution function using Numba for optimization.
    Parameters:
    coordinates (numpy.ndarray): Array of atomic coordinates.
    grid (numpy.ndarray): Grid data.
    midpoints (numpy.ndarray): Midpoints of the grid.

    Returns:
    tuple: A tuple containing the radial distribution function (numpy.ndarray) and the bin width (float).
    """
    remainder = grid.shape[0] % nworkers

    tasks_per_worker = [base_tasks for _ in range(nworkers)]

    for i in range(remainder):
        tasks_per_worker[i] += 1
    # print(grid.shape[0])
    bounds = np.array([0] + tasks_per_worker).cumsum()
    # print(bounds)

    origin = np.array((midpoints[0][0], midpoints[1][0], midpoints[2][0]))

    # make sure the bounding box is at least double the size of the grid
    box = np.array([midpoints[0][-1]*2, midpoints[1][-1]*2, midpoints[2][-1]*2, 90, 90, 90])
    
    # shift the coordinates and grid points to have an origin at (0,0,0)
    shift = coordinates.copy()
    shift -= origin
    for i in range(len(origin)):
        midpoints[i] -= origin[i]
    
    # ensure that the maximum distance fits within the box.  This is require FastNS
    nbins = int(min(max_cutoff / dr, min(box[:3]) / 2 / dr))

    # print(nbins, max_cutoff, max_cutoff / dr, min(box[:3]) / 2 , min(box[:3]) / 2 / dr)

    hist = np.zeros((nbins), dtype=np.int64)
    rdf = np.zeros(nbins)

    hist_all = np.zeros((nworkers,nbins), dtype=np.int64)
    rdf_all = np.zeros((nworkers, nbins))

    # for iworker, istart, istop in zip(range(nworkers), bounds[:-1], bounds[1:]):
    #     rdf_all[iworker,:], hist_all[iworker,:] = compute_rad_dist_numba(shift, grid, midpoints, dr, nbins, box, istart, istop)

    with multiprocessing.Pool() as p:
        results = p.map(
            worker, 
            [(shift, grid, midpoints, dr, nbins, box, iworker, istart, istop)
             for iworker, istart, istop in zip(range(nworkers), bounds[:-1], bounds[1:])])

    for i, (rdf, hist) in enumerate(results):
        rdf_all[i,:] = rdf
        hist_all[i,:] = hist

    # print(hist_all)
    # print(rdf_all)
    hist = hist_all.sum(axis=0)
    rdf = rdf_all.sum(axis=0)/hist
    return rdf, dr

def worker(args):
    shift, grid, midpoints, dr, nbins, box, iworker, istart, istop = args
    return compute_rad_dist_numba(shift, grid, midpoints, dr, nbins, box, istart, istop)


#@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
#@numba.jit(cache=True,forceobj=True, looplift=False, nogil=False,parallel=False,fastmath=True)
def compute_rad_dist_numba(coordinates, grid, midpoints, dr, nbins, box, start, stop):
    
    hist = np.zeros((nbins), dtype=np.int64)
    rdf = np.zeros(nbins)

    # print("test hist and rdf arrays:", hist, rdf)
    # create the cell list
    cell_list = MDAnalysis.lib.nsgrid.FastNS(
        dr * nbins, coordinates.astype('float32'), 
        box,
        pbc=False)
    
    
    for ix in range(start, stop):
        for iy in range(grid.shape[1]):
            # print(ix, iy)
            for iz in range(grid.shape[2]):
                # Compute the midpoint of the current grid cell
                mdpnt = np.array([[midpoints[0][ix], midpoints[1][iy], midpoints[2][iz]]])
                
                #print("test xyz:", iz, iy, ix, grid[ix,iy,iz]) #will print g(r) value at the given point
                result = cell_list.search(mdpnt.astype('float32'))
                try:
                    mindist = result.get_pair_distances().min()
                    binnum = int(mindist / dr)     
                    #print("The bin number is:", binnum)
                    if nbins > binnum:
                        rdf[binnum] += grid[ix,iy,iz] #increment by the g(r) values
                        hist [binnum] += 1 
                except ValueError:
                    # if there are no atoms within the cutoff
                    # print(ix,iy,iz)
                    pass
                
            # if iy == 2: return
    # rdf = rdf / hist
    return rdf, hist

def compute_rad_dist_python(coordinates, grid):
    return rdf, dr  # Return the RDF and bin width

    """
    Compute the radial distribution function without using Numba for optimization.

    Parameters:
    coordinates (numpy.ndarray): Array of atomic coordinates.
    grid (gridData.Grid): Grid data object.

    Returns:
    pandas.DataFrame: DataFrame containing the radial distribution function.
    """
    nbins = 200  # Number of bins for the RDF
    dr = 0.1  # Bin width

    # Initialize histograms for the RDF calculation
    hist = np.zeros((nbins), dtype=np.int64)
    rdf = np.zeros(nbins)

    # Iterate over the grid points
    for ix in range(grid.grid.shape[0]):
        for iy in range(grid.grid.shape[1]):
            for iz in range(grid.grid.shape[2]):
                # Compute the midpoint of the current grid cell
                mdpnt = np.array([grid.midpoints[0][ix], grid.midpoints[1][iy], grid.midpoints[2][iz]])
                mindist = 999999.0  # Initialize the minimum distance with a large value
                # Iterate over the atomic coordinates
                for iatom in range(coordinates.shape[0]):
                    # Compute the distance between the atom and the grid midpoint
                    dist = np.linalg.norm(mdpnt - coordinates[iatom])
                    if dist < mindist:
                        mindist = dist  # Update the minimum distance
                # Determine the appropriate bin for the current distance
                binnum = int(mindist / dr)
                if nbins > binnum:
                    rdf[binnum] += grid.grid[ix, iy, iz]  # Increment RDF by the grid value
                    hist[binnum] += 1  # Increment the histogram

    rdf = rdf / hist  # Normalize the RDF by the histogram
    # Create a DataFrame for the results and return it
    df = pd.DataFrame({'sep': np.linspace(dr / 2, nbins * dr - dr / 2, nbins), 'rdf': rdf})
    df = df.fillna(0)  # Fill any NaN values with 0
    
    return df 

# Executes as a Python script
if __name__ == "__main__":
    main()
