import numpy as np
from stl import mesh

'''This script performs a ray-triangle intersection by first getting the point of intersection, Q, then it computes whether Q lies inside of a designated triangle A,B,C'''

# p is origin of the ray, V is the direction unit vector originating from point p, t scales V to reach the point on the plane Q, and A,B,C are the points of the triangle
def ray_intersects_triangle(p, V, A, B, C):
    # Compute the plane's normal vector
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC) # n is normal vector to the plane
    n = n / np.linalg.norm(n)
    d = np.dot(n, A)
    denom = np.dot(n, V)
    if (denom == 0):
        return np.inf, np.array([0,0,0])
    # Compute the scalar t for the intersection point
    print(" value of p is", p)
    print(" value of d is", d)
    print(" value of n is", n)
    print(" value of A dotted with p is", np.dot(A, p))
    print(" value of n dotted with V is", np.dot(n, V))
    
    t = (d - np.dot(n,p)) / np.dot(n, V)
    print(" value of t is", t)
    # Compute the intersection point Q
    Q = p + t * V

    return t,Q
    
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
def inside_outside(p, V, A, B, C):
    t,Q = ray_intersects_triangle(p,V, A, B, C)
    if  t >= 0 and not t == np.inf:
        return True and is_point_in_triangle(A,B,C,Q)
    else:
        return False

    


