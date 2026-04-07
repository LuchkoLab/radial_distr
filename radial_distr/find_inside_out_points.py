import numpy as np
from gridData import Grid
from stl import mesh
import MDAnalysis as mda
import parmed as pmd
import mrcfile 
import numba
import argparse
import time
import numba_progress 
import numpy
'''This module takes in .stl and .mrc or .dx data, and it determines whether points lie inside or outside of a surface created by a .stl file then outputs the data to a .dx file'''

def main(stl_file_path, dx_file_path, output_file_path):

    """
    Loads a 3D STL model and a .dx volumetric grid, labels grid points as inside or outside 
    the mesh, and exports a new labeled .dx file.

    Parameters:
    -----------
    stl_file_path : str
        Path to the STL file representing the 3D mesh.
    dx_file_path : str
        Path to the input .dx file containing the volumetric grid.

    Outputs:
    --------
    - Prints the time taken to label all grid points.
    - Writes a new .dx file with inside/outside labels to 'ribo_data/inside_out_labels.dx'.
    """
    print("Loading stl file", flush=True), 
    model_mesh = load_stl(stl_file_path)
    print("Loading dx", flush=True)
    gridpoints, grid = load_dx(dx_file_path)
    print("files loaded", flush=True)
    gridpoints = gridpoints.reshape(list(grid.grid.shape)+[3])
    print("Labeling Points", flush=True)
    starttime = time.perf_counter()
    with numba_progress.ProgressBar(total=gridpoints.shape[0]) as pbar:
        labels = label_points_in_mesh(gridpoints, model_mesh.vectors, pbar)
    endtime = time.perf_counter()
    labels = labels.reshape(grid.grid.shape)
    print(f"How much time it took {endtime - starttime}", flush=True)
    grid.grid = labels
    #this outputs the new .dx file into this directory ribo_data/inside_out_labels.dx
    print("Outputing to file", flush=True)
    grid.export(output_file_path)



def get_args():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing paths to STL file, DX file, and output file.
    """
    parser = argparse.ArgumentParser(prog='read_in_files.py', description="Load .stl and .dx files needed for read_in_files to do raycasting.")

    # Define expected command-line arguments
    parser.add_argument('--stl', help='path to stl file expected', required=True) #stl file required
    
    parser.add_argument('--dx', help='path to dx file',required=True) #dx or mrc file required

    parser.add_argument('--output_dx', help='path and name for output dx file',required=True) #path and name to be given to output dx file

    args = parser.parse_args()
    return args




def load_stl(stl_file_path):
    """
    Load a 3D mesh from an STL file.

    Parameters:
        stl_file_path (str): Path to the STL file.

    Returns:
        mesh.Mesh: The loaded mesh object containing vertices and facets.
    """
    #Read in .stl file
    model_mesh = mesh.Mesh.from_file(stl_file_path)

    #test 1  if file was loaded with print statement, we will try accessing the vertices of the mesh
    #print(ala_stl.vectors.shape)

    #Access the number of facets
#    print(f"Number of facets: {model_mesh.vectors.shape[0]}")

    #Access and print an array containing the normal vectors for each facet
    #print(ala_stl.normals)
    
    #Access the vectors/facets
    #print(ala_stl.vectors)
    
    #Print shape
    #print(ala_stl.normals.shape)
    
    return model_mesh
# Programmic test from ChatGBT to see if binary file is being read correctly
#def is_ascii_stl(filename):
#    with open(filename, 'rb') as f:
 #       header = f.read(80)
  #      try:
   #         header.decode('ascii')
    #        return header.lower().startswith(b'solid')
     #   except UnicodeDecodeError:
      #      return False
      
#print(is_ascii_stl("model.stl"))



# p is origin of the ray, V is the direction unit vector originating from point p, t scales V to reach the point on the plane Q, and A,B,C are the points of the triangle
@numba.jit(cache=False,nopython=True, nogil=True,parallel=False,fastmath=False)
def ray_intersects_triangle(p, V, A, B, C):
    """
    Check if a ray intersects a triangle and return the intersection parameter and point.

    Parameters:
        p (np.ndarray): Origin point of the ray [x, y, z].
        V (np.ndarray): Direction vector of the ray [vx, vy, vz].
        A, B, C (np.ndarray): Vertices of the triangle [x, y, z].

    Returns:
        tuple: (t, Q) where t is the intersection parameter (np.inf if no intersection), 
               and Q is the intersection point [x, y, z].
    """
    # Convert all to d type float 32
    p = p.astype(np.float32)
    V = V.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = C.astype(np.float32)
    # Compute the plane's normal vector
    AB = (B - A).astype(np.float32)
    AC = (C - A).astype(np.float32)
    n = np.cross(AB, AC).astype(np.float32) # n is normal vector to the plane
#    n = n / np.linalg.norm(n) This version was not working with numba
    norm_n = np.float32(np.sqrt(np.dot(n, n)))  # Manually compute the norm
    n = n / norm_n
    d = np.float32(np.dot(n, A))
    denom = np.float32(np.dot(n, V))
    if (denom == 0):
       return np.inf, np.array([0,0,0], dtype=np.float32)
     #Compute the scalar t for the intersection point
     #print(" value of p is", p)
     #print(" value of d is", d)
     #print(" value of n is", n)
     #print(" value of A dotted with p is", np.dot(A, p))
     #print(" value of n dotted with V is", np.dot(n, V))
    
    t = (d - np.dot(n,p)) / denom
    #print(" value of t is", t)
    #Compute the intersection point Q
    Q = p + t * V

    return t, Q
@numba.jit(cache=False,nopython=True, nogil=True,parallel=False,fastmath=False)
def is_point_in_triangle(A, B, C, Q):
    """
    Check if a point Q lies inside the triangle defined by points A, B, C using barycentric coordinates.

    Parameters:
        A, B, C (np.ndarray): Vertices of the triangle [x, y, z].
        Q (np.ndarray): Point to check [x, y, z].

    Returns:
        bool: True if Q is inside the triangle, False otherwise.
    """
    
    AB = B -A
    AC = C - A
    n = np.cross(AB, AC)
    # Test edge AB
    AB = B -A
    cross1 = np.cross(AB, Q - A)
    dot1 = np.dot(n, cross1)

    # Test edge BC
    BC = C - B
    cross2 = np.cross(BC, Q - B)
    dot2 = np.dot(n, cross2)

    # Test edge CA
    CA = A - C
    cross3 = np.cross(CA, Q - C)
    dot3 = np.dot(n, cross3)

    # Check if qqall the dot products have the same sign (either all positive or all negative)
    #if (dot1 >= 0 and dot2 >= 0 and dot3 >= 0): Commented out to test if next line helps by including slightly negative values due to numerical precision issues
    if (dot1 >= 1e-6 and dot2 >= 1e-6 and dot3 >= 1e-6):   return True  # Q is inside the triangle
    else:
        return False  # Q is outside the triangle

@numba.jit(cache=False,nopython=True, nogil=True,parallel=False,fastmath=False)
def inside_outside(gridpoint, V, A, B, C):
    """
    Determine if a point is inside or outside a triangle based on ray intersection.

    Parameters:
        gridpoint (np.ndarray): The point to check [x, y, z].
        V (np.ndarray): Ray direction vector [vx, vy, vz].
        A, B, C (np.ndarray): Vertices of the triangle [x, y, z].

    Returns:
        bool: True if the point is inside, False otherwise.
    """
    
    t,Q = ray_intersects_triangle(gridpoint,V, A, B, C)
    if t >= 0 and t != np.inf: #line 1 added to correct parallel side
        return is_point_in_triangle(A,B,C,Q) #line 2 added to correct parallel side    
    #if  t >= 0 and not t == np.inf: line 1 removed to correct parallel side
        #return True and is_point_in_triangle(A,B,C,Q) line 2 removed to correct parallel side
    #else: line 3 removed to correct parallel side
    
    # "Line" 3 block of if statements for coplanar rays
    if ray_intersects_triangle(p,V,A,B,C):
        if ray_intersects_triangle(p,V,A,B): return True
        if ray_intersects_triangle(p,V,B,C): return True
        if ray_intersects_triangle(p,V,C,A): return True
    return False

def load_dx(dx_file_path):
    """
    Load a volumetric grid from a DX file.

    Parameters:
        dx_file_path (str): Path to the DX file.

    Returns:
        tuple: (midpoints, grid) where midpoints is a numpy array of grid point coordinates,
               and grid is the Grid object.
    """
    #Read in dx file
    grid = Grid(dx_file_path)

    #Access the grid data
    data = grid.grid

    #test if data was loaded by getting grid shape and origin
    grid_shape = grid.grid.shape
    origin = grid.origin
    midpoints = grid.midpoints
    edges = grid.edges
    delta = grid.delta
#    print("Grid Shape:", grid_shape)
#    print("Origin:", origin)
#    print("Midpoints:", midpoints)
#    print("Edges:", edges)
#    print("D(x) value:", delta)    #Iterate through and print grid values
    
    #loop through all midpoints
    midpoints = []
    for ix in range(grid.grid.shape[0]):
       for iy in range(grid.grid.shape[1]):
            for iz in range(grid.grid.shape[2]):
                # Access the midpoint for the current grid point
                midpoint = [grid.midpoints[0][ix], grid.midpoints[1][iy], grid.midpoints[2][iz]]
                #print(f"Midpoint at ({ix}, {iy}, {iz}): {midpoint}")
                midpoints.append(midpoint)
                #print(f"Midpoint at ({ix}, {iy}, {iz}): {midpoint}")
    #print(midpoints)            
    

    return np.array(midpoints), grid
@numba.jit(cache=False,nopython=True, nogil=True,parallel=False,fastmath=False)
def is_point_near_facet(gridpoint, facet):
    """
    Check if a grid point (x, y, z) is near a facet in 3D space.

    Args:
        gridpoint: The 3D coordinate of the grid point [x, y, z].
        facet: A list of three vertices defining the facet (triangle) in 3D.

    Returns:
        bool: True if the gridpoint is near any facet in the x, y, or z direction. False if not.
    """
    # Extract the facet's x, y, z coordinates
    A, B, C = facet  

    # Get min/max values for x, y from the facet's vertices
    min_x, max_x = min(A[0], B[0], C[0]), max(A[0], B[0], C[0])
    min_y, max_y = min(A[1], B[1], C[1]), max(A[1], B[1], C[1])


    # Get gridpoint (x, y, z)
    x, y, z = gridpoint

    # If the gridpoint is completely outside the facet's x, y, or z bounds, exclude it
    if (
        x < min_x or x > max_x or 
        y < min_y or y > max_y):
        return False  # Exclude this gridpoint if no facet is near

    return True  # Keep gridpoint if it's within the facet's bounds

def filter_all_grid_slices_near_facets(grid, facets):
    """
    Filters grid points from all Z-slices that are near any facet.

    Args:
        grid (np.ndarray): 4D array of grid points (Nx, Ny, Nz, 3).
        facets (list): List of triangle facets.

    Returns:
        list: Filtered grid points across all slices.
    """
    filtered_points = []

    for z_index in range(grid.shape[2]):  # Loop through all Z-layers
        for ix in range(grid.shape[0]):
            for iy in range(grid.shape[1]):
                gridpoint = grid[ix, iy, z_index]

                for facet in facets:
                    if is_point_near_facet(gridpoint, facet):
                        filtered_points.append(gridpoint)
                        break

    return filtered_points

def check_filtered_points(filtered_points, facets):
    """
    Check each filtered point to determine if it is inside or outside the mesh.

    Parameters:
        filtered_points (list): List of grid points near facets.
        facets (list): List of triangle facets defining the mesh.

    Returns:
        list: List of tuples (point, inside, t_vals) where inside is bool and t_vals are intersection parameters.
    """

    checked_points = []

    for point in filtered_points:
        inside, t_vals = is_point_inside_mesh(point, facets)
    checked_points.append((point, inside, t_vals))
    return checked_points



        
# Function to check if a point is inside the mesh using ray casting
@numba.jit(cache=False,nopython=True, nogil=True,parallel=False,fastmath=False)
def is_point_inside_mesh(gridpoint, facets):
    """
    Determine if a point is inside a mesh using ray casting algorithm.

    Parameters:
        gridpoint (np.ndarray): The point to check [x, y, z].
        facets (list): List of triangle facets, each as [A, B, C] where A, B, C are vertices.

    Returns:
        tuple: (inside, t_vals) where inside is bool indicating if point is inside,
               and t_vals is array of intersection parameters.
    """
    
    ray_direction = np.array([0.0, 0.0, 1.0])  # Arbitrary ray direction
    intersections = 0
    t_vals = np.zeros(len(facets),dtype=np.float32)
    
#    Q_vals = np.zeros(len(facets),dtype=np.float32)
    for facet in facets:
        if not is_point_near_facet(gridpoint, facet):
            continue  # Skip this facet if the gridpoint is not near it
        A, B, C = facet
        
        t, Q = ray_intersects_triangle(gridpoint, ray_direction, A, B, C)
        print("gridpoint:", gridpoint)
        print("t value is", t)
        print("is_point_in_triangle values", is_point_in_triangle(A, B, C, Q))
        print("Facet vertices")
        print("  A:", A)
        print("  B:", B)
        print("  C:", C)
        print("Q is :", Q)
        if t >= 0 and is_point_in_triangle(A, B, C, Q):
            t_vals[intersections] = t
            
           
            intersections += 1
    
    t_vals = t_vals[0:intersections]
    #print("t_vals", t_vals)

    # remove duplicate intersections caused by shared edges/vertices
    if intersections > 1:

        t_vals = np.sort(t_vals)

    unique = [t_vals[0]]

    for t in t_vals[1:]:
        if abs(t - unique[-1]) > 1e-6:
            unique.append(t)

    t_vals = np.array(unique)
    intersections = len(t_vals)
    
    return intersections % 2 == 1, t_vals

#Function to label points inside or outside the mesh
@numba.jit(cache=False,nopython=True, nogil=True,parallel=True,fastmath=False)
def label_points_in_mesh(points, facets, pbar=None):
    """
    Label all grid points as inside (0) or outside (1) the mesh using ray casting.

    Parameters:
        points (np.ndarray): 4D array of grid points (Nx, Ny, Nz, 3).
        facets (list): List of triangle facets defining the mesh.
        pbar (numba_progress.ProgressBar, optional): Progress bar for tracking.

    Returns:
        np.ndarray: 3D array of labels (0 for inside, 1 for outside).
    """
   
  
    labels = np.ones((points.shape[0], points.shape[1], points.shape[2]))  # Initialize all points as outside (1)
    test_indices = [(127,225),(132,203),(283,104),(284,104)]

    for ix, iy in test_indices:

        point = points[ix, iy]

        inside, t_vals = is_point_inside_mesh(points[ix, iy, 0], facets)

        print("Testing:", ix, iy)
        print("Point:", points[ix,iy,0])
        print("inside:", inside)
        print("t_vals:", t_vals)
        print("Number of t_vals:", len(t_vals))
        print("Number of facets:", len(facets))
        print("Number of intersections:", len(t_vals)
                                       )
    #print("points.shape", points.shape)
    #Unccommented to test indices (1)
    # for ix in numba.prange(points.shape[0]):
    #uncomment next line to test for a single point in the middle of the grid
    #for ix in range(int(points.shape[0]/2),int(points.shape[0]/2 +1)):
        #print("ix", ix)
        #Uncommented to test indices (2)
        # for iy in range(points.shape[1]):
            #uncomment next line to test for a single point in the middle of the grid
        #for iy in range(int(points.shape[1]/2),int(points.shape[1]/2 +1)):
            #Uncommented to test indices (3)
            # point = points[ix, iy]
            #print("len(facets)", len(facets))
            #Uncommented to tests indices (4)
            # inside, t_vals = is_point_inside_mesh(points[ix, iy, 0], facets)
            #print("inside and t_vals", inside, t_vals)
            #break
            #Uncommented to tests indices (5)
            # if t_vals.size == 0:
                #Uncommentted to test indices (6)
                # labels[ix, iy] = 1  # Outside
        continue 
           
            #print("======")
            #print (points[ix, iy, 0], t_vals)
#if t_vals is an empty array, then that ray has zero intersections,stay outside

    for iz in range(1,points.shape[2]):
            
                i_t_vals = t_vals - (points[ix,iy, iz,2] - points[ix, iy,0,2])
                n_pos = np.sum(i_t_vals > 0)
                n_neg = np.sum(i_t_vals < 0)   
                if n_pos % 2 == 0 and n_neg % 2 == 0 and (n_pos + n_neg) > 0:
                    labels[ix, iy, iz] = 1
                         
                else:
                        
                    labels[ix,iy,iz] = 0   
            # uncomment this line to check for a single point    
            #break
        #if pbar is not None:
            #pbar.update(1)
    return labels

# Run the script with example files
if __name__ == "__main__":
    args = get_args()
    main(args.stl, args.dx, args.output_dx)
   # filtered_points = filter_all_grid_slices_near_facets(grid, facets)
