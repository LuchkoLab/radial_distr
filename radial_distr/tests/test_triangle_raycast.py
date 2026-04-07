import pytest
import radial_distr.triangle_raycast as triangle_raycast
import sys
import numpy as np
import radial_distr.find_inside_out_points as find_inside_out_points

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


def test_point_inside_bounds():
    """
    Tests that a point within a triangle's bounding box returns True.
    Input:
    - A triangle and a point that is inside of the bounding box
    Expected Output:
    - The function returns 'True' because the point is inside of the bounding box
    """
    point = np.array([2, 2, 0])
    facet = [
        np.array([1, 1, 0]),
        np.array([3, 1, 0]),
        np.array([2, 3, 0])
    ]
    assert find_inside_out_points.is_point_near_facet(point, facet) == True

def test_point_outside_bounds():
    """
    Tests that a point outside a triangle's bounding box returns False.
    Input:
    - A triangle and a point outside the bounding box
    Expected output
    - The function reeturns 'False' because the point is outside of the bounding box
    """
    point = np.array([5, 5, 0])
    facet = [
        np.array([1, 1, 0]),
        np.array([3, 1, 0]),
        np.array([2, 3, 0])
    ]
    assert find_inside_out_points.is_point_near_facet(point, facet) == False


def test_watertight_shared_edge_counts_once_after_dedup():
    point = np.array([-160.46308899, -84.86076355, -163.58261108], dtype=np.float64)
    facets = np.array([
        [[-161.037, -84.789, -86.835], [-160.082, -84.797, -87.004], [-160.391, -85.118, -85.876]],
        [[-160.202, -84.724, -62.716], [-160.978, -84.527, -63.682], [-160.199, -85.554, -63.874]],
        [[-159.877, -85.864, -31.849], [-160.71, -84.707, -31.841], [-160.054, -84.715, -32.979]],
        [[-160.132, -84.766, 5.145], [-161.285, -84.274, 5.141], [-161.285, -85.096, 3.99]],
        [[-160.132, -84.766, 5.145], [-161.285, -85.096, 3.99], [-160.135, -85.424, 3.991]],
    ], dtype=np.float64)

    inside, t_vals = find_inside_out_points.is_point_inside_mesh(point, facets)

    assert inside is False
    np.testing.assert_allclose(
        t_vals,
        np.array([76.86813754, 100.25713636, 131.50218133, 168.39594321]),
        atol=1e-5,
    )


def test_watertight_multiple_shared_triangles_stay_even():
    point = np.array([-84.96308899, -134.36076355, -163.58261108], dtype=np.float64)
    facets = np.array([
        [[-85.248, -134.237, -16.834], [-84.106, -134.074, -15.693], [-85.252, -134.402, -15.688]],
        [[-85.252, -134.402, -15.688], [-84.106, -134.074, -15.693], [-84.11, -134.239, -14.547]],
        [[-85.252, -134.402, -15.688], [-84.11, -134.239, -14.547], [-85.246, -134.724, -14.554]],
        [[-85.246, -134.724, -14.554], [-84.11, -134.239, -14.547], [-84.104, -134.561, -13.412]],
        [[-83.941, -134.193, 21.402], [-85.212, -135.324, 21.26], [-84.072, -135.332, 20.111]],
        [[-85.223, -134.182, 22.412], [-85.212, -135.324, 21.26], [-83.941, -134.193, 21.402]],
    ], dtype=np.float64)

    inside, t_vals = find_inside_out_points.is_point_inside_mesh(point, facets)

    assert inside is False
    np.testing.assert_allclose(
        t_vals,
        np.array([148.18327054, 185.61309077]),
        atol=1e-5,
    )
