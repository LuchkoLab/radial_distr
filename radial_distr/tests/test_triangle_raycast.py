import pytest
import radial_distr.triangle_raycast as triangle_raycast
import sys
import numpy as np

def test_triangle_raycast_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "radial_distr.triangle_raycast" in sys.modules

def test_inside():
    """From triangle raycast, call your function"""
    inside = triangle_raycast.inside_outside(np.array([0,0,0]),np.array([1,0,0]),np.array([1,-1,-1]),np.array([1,-1,1]),np.array([1,1,0]))
    assert inside
    
def test_ray_on_line():
    """Test if ray along a facet edge is not inside"""
    same_line = triangle_raycast.inside_outside(np.array([1,2,0]),np.array([3,4,0]),np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert same_line == False

def test_ray_parallel():
    """Test if ray is not parallel to plane """
    parallel_to_plane = triangle_raycast.inside_outside(np.array([0,0,0]),np.array([2,-1,0]),np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert parallel_to_plane == False

def test_ray_intersects_vertex():
    """Tests if ray does not  intersect a vertex of the triangle"""
    ray_intersects_a_vertex = triangle_raycast.inside_outside(np.array([0,0,0]),np.array([1,2,0]),np.array([1,2,0]), np.array([4,6,0]),np.array([2,4,0]))
    assert ray_intersects_a_vertex == False

def test_if_t_is_positive():
    """Test if t is pointing away from the plane"""
    ray_points_away_from_plane = triangle_raycast.inside_outside(np.array([0,0,0]), np.array([0,0,-2]), np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert ray_points_away_from_plane == False

def test_t_not_in_triangle():
    """Test if t is positive but does not intersect any vertices or does not intersect the triangle"""
    outside = triangle_raycast.inside_outside(np.array([0,0,0]), np.array([1,3,0]), np.array([1,1,1]), np.array([1,-1,1]), np.array([1,0,2]))
    assert outside == False 
