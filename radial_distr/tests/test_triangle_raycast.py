import pytest
import radial_distr.triangle_raycast as triangle_raycast
import sys
import numpy as np

def test_triangle_raycast_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "radial_distr.triangle_raycast" in sys.modules

def test_inside():
    """From triangle raycast, call your function to check if the function correctly identifies a point as being inside a triangle
    Inputs:
        - A ray origin point and vector (p, V)
        - vertices of a triangle defining a plane (A,B,C)
    Expected Output:
        - The function returns 'true' for points in the triangle """
    inside = triangle_raycast.inside_outside(np.array([0,0.5,1]),np.array([0,0,-1]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert inside
    
def test_ray_on_line():
    """Test if a ray that is along the same line as a facet edge does not count as being inside of the triangle.
    Inputs:
    - A ray and origin that lie on one edge of the triangle
    Output:
    - The function returns 'False' because the point lies on the edge and not inside the triangle """
    same_line = triangle_raycast.inside_outside(np.array([-1,0,0]),np.array([2,0,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert same_line == False

def test_ray_parallel():
    """Test that a ray parallel to the triangle plane does not return as intersecting the triangle
        Inputs:
        - a ray parallel to the triangle plane
        
        Expected output:
        - The function returns 'False' because the ray does not intersect the triangle. """
    parallel_to_plane = triangle_raycast.inside_outside(np.array([0,0,1]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert parallel_to_plane == False

def test_ray_intersects_vertex():
    """Tests if ray intersects a vertex of the triangle which is part of the triangle, it will return True (I have questions about this...)
    Input:
    - a ray that intersects one of the vertices of the triangle 
    Output:
    - the function returns ??'True' because it is not entering the interior of the triangle Perhaps we can return a number which indicates how many triangles it belongs to"""
    ray_intersects_a_vertex = triangle_raycast.inside_outside(np.array([0,0,-1]),np.array([0,1,1]),np.array([-1,0,0]), np.array([0,1,0]),np.array([1,0,0]))
    assert ray_intersects_a_vertex == True

def test_if_t_is_positive():
    """Tests that if t is pointing away from the plane that the ray does not count as intersecting the triangle.
    Inputs:
    - A ray that points away from the triangle's plane.
    Expected Output
    - The function returns 'False' because the ray points in the opposite direction of the plane. """
    ray_points_away_from_plane = triangle_raycast.inside_outside(np.array([0,0,1]), np.array([0,0,1]), np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert ray_points_away_from_plane == False

def test_t_not_in_triangle():
    """Test if t is positive but does not intersect any vertices or does not intersect the triangle returns 'False'
    Input:
    - A ray that passes near the triangle plane, but does not intersect triangle
    Expected Output:
    - The function returns 'False' because the intersection points lies outside of the triangle """
    outside = triangle_raycast.inside_outside(np.array([1,0,0.5]), np.array([0,0,1]), np.array([-1,0,0]), np.array([0,1,0]), np.array([1,0,0]))
    assert outside == False 
