#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import numpy as np
import vocd

def test_tetrahedrization():
    """Test 3D Delaunay tetrahedrization"""
    print("Testing 3D tetrahedrization...")

    verts, tris = vocd.create_cube(2.0, 3.0, 4.0)

    verts = np.array(verts, dtype=np.float64).reshape(-1, 3)
    tris  = np.array( tris, dtype=np. uint32).reshape(-1, 3)
    
    # Triangulate
    triangles = vocd.tetrahedrize(verts, tris)
    
    print(f"From {verts.shape} points")
    print(f"Tets: {triangles}")
    print()

def test_manifold():
    """Test Manifold cube creation"""
    print("Testing Manifold...")
    
    # Create a cube
    cube_verts, cube_tris = vocd.create_cube(2.0, 3.0, 4.0)
    
    print(f"Created cube with dimensions 2x3x4")
    print(f"Cube verts: {cube_verts}")
    print(f"Cube verts: {cube_tris}")
    print()

def test_voronoi():
    """Test 3D Voronoi diagram"""
    print("Testing 3D Voronoi...")
    
    # Create some random 3D points
    points = np.random.rand(20, 3) * 10
    wts = np.random.rand(20)
    
    # Define bounds
    bounds = np.array([0, 10, 0, 10, 0, 10], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # Compute Voronoi cell volumes
    cells = vocd.voronoi_3d(points, wts, bounds)
    
    print(f"Computed Voronoi cells for {points.shape} points")
    print(f"Voronoi cell hull vertices: {cells}")
    #print(f"Computed {len(volumes)} Voronoi cell volumes")
    #print(f"Average volume: {np.mean(volumes):.4f}")
    #print(f"Min volume: {np.min(volumes):.4f}, Max volume: {np.max(volumes):.4f}")
    print()

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    test_tetrahedrization()
    test_manifold()
    test_voronoi()
