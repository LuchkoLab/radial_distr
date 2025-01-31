import numpy as np
from gridData import Grid
from stl import mesh
import MDAnalysis as mda
import parmed as pmd
import mrcfile 
import numba
import argparse

'''This module takes in .stl and .mrc or .dx data, and it determines whether points lie inside or outside of a surface created by a .stl file '''

# Main function to run the script
def main(stl_file_path, dx_file_path):
    
    model_mesh = load_stl(stl_file_path)
    load_dx(dx_file_path)

    gridpoints, grid = load_dx(dx_file_path)
    gridpoints = gridpoints.reshape(list(grid.grid.shape)+[3])
    labels = label_points_in_mesh(gridpoints, model_mesh.vectors)
    labels = labels.reshape(grid.grid.shape)
    grid.grid = labels
    grid.export("tests/data/inside_out_labels.dx")



def get_args():
    parser = argparse.ArgumentParser(prog='read_in_files.py', description="Load .stl and .dx files needed for read_in_files to do raycasting.")

    # Define expected command-line arguments
    parser.add_argument('--stl', help='path to stl file expected', required=True)
    #need to add argument for user to give name
    parser.add_argument('--dx', help='path to dx file',required=True)
    args = parser.parse_args()
    return args




def load_stl(stl_file_path):
    #Read in .stl file
    model_mesh = mesh.Mesh.from_file(stl_file_path)

    #test if file was loaded with print statement, we will try accessing the vertices of the mesh
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

# p is origin of the ray, V is the direction unit vector originating from point p, t scales V to reach the point on the plane Q, and A,B,C are the points of the triangle
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def ray_intersects_triangle(p, V, A, B, C):
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
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def is_point_in_triangle(A, B, C, Q):
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

    # Check if all the dot products have the same sign (either all positive or all negative)
    if (dot1 >= 0 and dot2 >= 0 and dot3 >= 0):
        return True  # Q is inside the triangle
    else:
        return False  # Q is outside the triangle
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def inside_outside(gridpoint, V, A, B, C):
    t,Q = ray_intersects_triangle(p,V, A, B, C)
    if  t >= 0 and not t == np.inf:
        return True and is_point_in_triangle(A,B,C,Q)
    else:
        return False

def load_dx(dx_file_path):
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

# Function to check if a point is inside the mesh using ray casting
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def is_point_inside_mesh(gridpoint, facets):
    
    ray_direction = np.array([0.0, 0.0, 1.0])  # Arbitrary ray direction
    intersections = 0
    t_vals = [] 
#    Q_vals = np.zeros(len(facets),dtype=np.float32)
    for facet in facets:
        A, B, C = facet
        t, Q = ray_intersects_triangle(gridpoint, ray_direction, A, B, C)
        if t >= 0 and is_point_in_triangle(A, B, C, Q):
            
            t_vals.append(t)
#            Q_vals[intersections] = Q
            intersections += 1
    
    
    
    
    return intersections % 2 == 1, np.array(t_vals)

#Function to label points inside or outside the mesh
@numba.jit(cache=True,nopython=True, nogil=True,parallel=True,fastmath=True)
def label_points_in_mesh(points, facets):
    labels = np.ones((points.shape[0], points.shape[1], points.shape[2]))  # Initialize all points as outside (1)
    
    
    for ix in numba.prange(points.shape[0]):
        for iy in range(points.shape[1]):
            point = points[ix, iy]
            inside, t_vals = is_point_inside_mesh(points[ix, iy, 0], facets)
            if t_vals.size == 0:
                labels[ix, iy] = 1  # Outside
                continue 
           
           # print("======")
           # print (points[ix, iy, 0], t_vals)
#if t_vals is an empty array, then that ray has zero intersections,stay outside

            for iz in range(1,points.shape[2]):
            
                i_t_vals = t_vals - (points[ix,iy, iz,2] - points[ix, iy,0,2])
                n_pos = np.sum(i_t_vals > 0)
                n_neg = np.sum(i_t_vals < 0)   
                if n_pos % 2 == 0 and n_neg % 2 == 0 and (n_pos + n_neg) > 0:
                    labels[ix, iy, iz] = 1
                         
                else:
                        
                    labels[ix,iy,iz] = 0   
    
        
    return labels

# Run the script with example files
if __name__ == "__main__":
    #stl_file = '/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.stl'
    #dx_file = '/home/tyork/ribosome/radial_distr/radial_distr/tests/guv.O.5.dx'
    args = get_args()
    main(args.stl, args.dx)

