import pytest
import radial_distr.triangle_raycast as triangle_raycast
import sys
import numpy as np

def test_triangle_raycast_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "radial_distr.triangle_raycast" in sys.modules

def test_inside():
    """From triangle raycast, call your function"""
    inside = triangle_raycast.inside_outside(np.array([0,0.5,1]),np.array([0,0,-1]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert inside
    
def test_ray_on_line():
    """Test if ray intersects an edge/facet of triangle which is part of the triangle, it will return true"""
    same_line = triangle_raycast.inside_outside(np.array([0,0,1]),np.array([0,0,-1]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert same_line == True

def test_ray_parallel():
    """Test if ray is not parallel to plane """
    parallel_to_plane = triangle_raycast.inside_outside(np.array([0,0,1]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert parallel_to_plane == False

def test_ray_intersects_vertex():
    """Tests if ray intersects a vertex of the triangle which is part of the triangle, it will return True"""
    ray_intersects_a_vertex = triangle_raycast.inside_outside(np.array([1,0,1]),np.array([0,0,-1]),np.array([-1,0,0]), np.array([0,1,0]),np.array([1,0,0]))
    assert ray_intersects_a_vertex == True

def test_if_t_is_positive():
    """Test if t is pointing away from the plane"""
    ray_points_away_from_plane = triangle_raycast.inside_outside(np.array([0,0,1]), np.array([0,0,1]), np.array([-1,0,0]),np.array([0,1,0]),np.array([1,0,0]))
    assert ray_points_away_from_plane == False

def test_t_not_in_triangle():
    """Test if t is positive but does not intersect any vertices or does not intersect the triangle"""
    outside = triangle_raycast.inside_outside(np.array([1,0,0.5]), np.array([0,0,1]), np.array([-1,0,0]), np.array([0,1,0]), np.array([1,0,0]))
    assert outside == False 
