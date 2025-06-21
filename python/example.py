#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import numpy as np
import vocd

def test_triangulation():
    """Test 2D Delaunay triangulation"""
    print("Testing 2D Triangulation...")
    
    # Create some random 2D points
    points = np.random.rand(10, 2)
    
    # Triangulate
    triangles = vocd.tetrahedrize(points)
    
    print(f"Generated {len(triangles)} triangles from {len(points)} points")
    print(f"First triangle indices: {triangles[0]}")
    print()

def test_manifold():
    """Test Manifold cube creation"""
    print("Testing Manifold...")
    
    # Create a cube
    cube = vocd.create_cube(2.0, 3.0, 4.0)
    
    print(f"Created cube with dimensions 2x3x4")
    print(f"Cube object: {cube}")
    print()

def test_voronoi():
    """Test 3D Voronoi diagram"""
    print("Testing 3D Voronoi...")
    
    # Create some random 3D points
    points = np.random.rand(20, 3) * 10
    
    # Define bounds
    bounds = [0, 10, 0, 10, 0, 10]  # [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # Compute Voronoi cell volumes
    volumes = vocd.voronoi_3d(points, bounds)
    
    print(f"Computed {len(volumes)} Voronoi cell volumes")
    print(f"Average volume: {np.mean(volumes):.4f}")
    print(f"Min volume: {np.min(volumes):.4f}, Max volume: {np.max(volumes):.4f}")
    print()

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    #est_triangulation()
    test_manifold()
    #test_voronoi()