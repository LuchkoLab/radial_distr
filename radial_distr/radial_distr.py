import argparse
import numpy as np
import parmed as pmd
import pandas as pd
import gridData
import numba

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
    parm = pmd.load_file(args.p, args.y)
    print("test print", parm.coordinates)  # Print the molecular coordinates
    print(parm.coordinates.shape)  # Print the shape of the coordinates array
    
    # Compute the radial distribution function using Numba optimization
    rdf, dr = compute_rad_dist_numba(parm.coordinates, g.grid, g.midpoints)
    # Create a DataFrame for the results and save it to a CSV file
    df = pd.DataFrame({'sep': np.linspace(dr / 2, len(rdf) * dr - dr / 2, len(rdf)), 'rdf': rdf})
    df = df.fillna(0)  # Fill any NaN values with 0
    print(df)
    df.to_csv('rdf.csv', index=False)  # Save the DataFrame to a CSV file

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
    parser.add_argument('-p', help='parmtop file', required=True)
    parser.add_argument('-y', help='coordinate file', required=True)
    # Parse and return the arguments
    args = parser.parse_args()
    return args

@numba.jit(cache=True, nopython=True, nogil=True, parallel=False, fastmath=True)
def compute_rad_dist_numba(coordinates, grid, midpoints):
    """
    Compute the radial distribution function using Numba for optimization.

    Parameters:
    coordinates (numpy.ndarray): Array of atomic coordinates.
    grid (numpy.ndarray): Grid data.
    midpoints (numpy.ndarray): Midpoints of the grid.

    Returns:
    tuple: A tuple containing the radial distribution function (numpy.ndarray) and the bin width (float).
    """
    nbins = 200  # Number of bins for the RDF
    dr = 0.1  # Bin width

    # Initialize histograms for the RDF calculation
    hist = np.zeros((nbins), dtype=np.int64)
    rdf = np.zeros(nbins)
    print("test hist and rdf arrays:", hist, rdf)  # Print initialized arrays for debugging

    # Iterate over the grid points using Numba for optimization
    for ix in numba.prange(grid.shape[0]):
        for iy in range(grid.shape[1]):
            print(iy)
            for iz in range(grid.shape[2]):
                # Compute the midpoint of the current grid cell
                mdpnt = np.array([midpoints[0][ix], midpoints[1][iy], midpoints[2][iz]])
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
                    rdf[binnum] += grid[ix, iy, iz]  # Increment RDF by the grid value
                    hist[binnum] += 1  # Increment the histogram
            if iy == 2:  # Exit the loop early for testing purposes
                return rdf, dr

    rdf = rdf / hist  # Normalize the RDF by the histogram
    return rdf, dr  # Return the RDF and bin width

def compute_rad_dist(coordinates, grid):
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
        break  # Only process the first slice of iy for demonstration

    rdf = rdf / hist  # Normalize the RDF by the histogram
    # Create a DataFrame for the results and return it
    df = pd.DataFrame({'sep': np.linspace(dr / 2, nbins * dr - dr / 2, nbins), 'rdf': rdf})
    df = df.fillna(0)  # Fill any NaN values with 0
    
    return df 

# Executes as a Python script
if __name__ == "__main__":
    main()
