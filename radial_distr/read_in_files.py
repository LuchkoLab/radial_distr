import numpy as np
from gridData import Grid
from stl import mesh
import MDAnalysis as mda
import parmed as pmd
import mrcfile 
import numba

'''This module takes in .stl and .mrc or .dx data, and it determines whether points lie inside or outside of a surface created by a .stl file '''


def load_stl():
    #Read in .stl file
    model_mesh = mesh.Mesh.from_file('/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.stl')

    #test if file was loaded with print statement, we will try accessing the vertices of the mesh
    #print(ala_stl.vectors.shape)

    #Access the number of facets
    print(f"Number of facets: {model_mesh.vectors.shape[0]}")

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

    return t,Q
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
    print("Grid Shape:", grid_shape)
    print("Origin:", origin)
    print("Midpoints:", midpoints)
    print("Edges:", edges)
    print("D(x) value:", delta)    #Iterate through and print grid values
    
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
    

    return np.array(midpoints)

# Function to check if a point is inside the mesh using ray casting
@numba.jit(cache=True,nopython=True, nogil=True,parallel=False,fastmath=True)
def is_point_inside_mesh(gridpoint, model_mesh):
    
    ray_direction = np.array([1.0, 0.0, 0.0])  # Arbitrary ray direction
    intersections = 0
    for facet in model_mesh.vectors:
        A, B, C = facet
        t, Q = ray_intersects_triangle(gridpoint, ray_direction, A, B, C)
        if t >= 0 and is_point_in_triangle(A, B, C, Q):
            intersections += 1
    # Point is inside the mesh if the number of intersections is odd
    return intersections % 2 == 1

#Function to label points inside or outside the mesh
def label_points_in_mesh(points, model_mesh):
    labels = np.ones(len(points))  # Initialize all points as outside (1)
    
    for i, point in enumerate(points):
    
        if is_point_inside_mesh(point, model_mesh):
            labels[i] = 0  # Label as 0 if inside the mesh
            
    return labels


# Main function to run the script
def main(stl_file_path, dx_file_path):
    model_mesh = load_stl()
    #print("The facets are", facets)
    for gridpoint in load_dx(dx_file_path):
        print(gridpoint)
        inside = is_point_inside_mesh(gridpoint, model_mesh)
        break




# Run the script with example files
if __name__ == "__main__":
    stl_file = '/home/tyork/ribosome/radial_distr/radial_distr/tests/ala.stl'
    dx_file = '/home/tyork/ribosome/radial_distr/radial_distr/tests/guv.O.5.dx'
    main(stl_file, dx_file)

