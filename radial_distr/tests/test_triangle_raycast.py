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
    """Test if ray is on same line as facet"""
    same_line = triangle_raycast.inside_outside(np.array([1,2,0]),np.array([3,4,0]),np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert same_line

def test_ray_parallel():
    """Test is ray is parallel to plane """
    parallel_to_plane = triangle_raycast.inside_outside(np.array([0,0,0]),np.array([2,-1,0]),np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert parallel_to_plane

def test_ray_intersects_vertex():
    """Test if ray intersects a vertex"""
    ray_intersects_a_vertex = triangle_raycast.inside_outside(np.array([0,0,0]),np.array([1,2,0]),np.array([1,2,0]), np.array([4,6,0]),np.array([2,4,0]))
    assert ray_intersects_a_vertex

def test_if_t_is_positive():
    """Test if t is positive and pointing in the right direction"""
    ray_points_away_from_plane = triangle_raycast.inside_outside(np.array([0,0,0]), np.array([0,0,-2]), np.array([1,2,0]),np.array([4,6,0]),np.array([2,4,0]))
    assert ray_points_away_from_plane
