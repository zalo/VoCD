#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import vocd
import trimesh
import numpy as np

def test_tetrahedrization():
    """Test 3D Delaunay tetrahedrization"""
    print("Testing 3D tetrahedrization...")

    verts, tris = vocd.create_sphere(1.0)#create_cube(2.0, 3.0, 4.0)

    verts = np.array(verts, dtype=np.float64).reshape(-1, 3)
    tris  = np.array( tris, dtype=np. uint32).reshape(-1, 3)
    
    # Triangulate
    tet_vertices, tet_indices = vocd.tetrahedrize(verts, tris)
    
    print(f"From {verts.shape} points")
    print(f"Tets: Vertices: {tet_vertices} and Tets: {tet_indices}")
    print()

    visualize_tetrahedra(tet_vertices, tet_indices)

def visualize_tetrahedra(tet_vertices, tet_indices):
    """Visualize tetrahedra using trimesh"""
    # Visualize tetrahedra with scaling
    tet_scale = 0.8  # Equivalent to simulationParams.tetScale
    
    # Process tetrahedra with scaling
    scaled_vertices = []
    vertex_colors = []
    
    for tet in tet_indices:
        # Get vertices for this tetrahedron
        v0 = np.array(tet_vertices[tet[0]])
        v1 = np.array(tet_vertices[tet[1]])
        v2 = np.array(tet_vertices[tet[2]])
        v3 = np.array(tet_vertices[tet[3]])
        
        # Calculate average position (centroid)
        avg_pos = (v0 + v1 + v2 + v3) / 4
        
        # Scale vertices around centroid
        v0_scaled = (v0 - avg_pos) * tet_scale + avg_pos
        v1_scaled = (v1 - avg_pos) * tet_scale + avg_pos
        v2_scaled = (v2 - avg_pos) * tet_scale + avg_pos
        v3_scaled = (v3 - avg_pos) * tet_scale + avg_pos
        
        # Check if tetrahedron is degenerate (coplanar)
        # Calculate volume using scalar triple product
        edge1 = v1_scaled - v0_scaled
        edge2 = v2_scaled - v0_scaled
        edge3 = v3_scaled - v0_scaled
        volume = np.abs(np.dot(edge1, np.cross(edge2, edge3))) / 6.0
        
        if volume < 0.001:
            continue
        
        # Generate random color for this tetrahedron
        color = np.random.rand(3)

        # Push the modified vertices into the array as four triangles
        scaled_vertices.extend([v0_scaled, v1_scaled, v2_scaled])
        scaled_vertices.extend([v2_scaled, v3_scaled, v0_scaled])
        scaled_vertices.extend([v3_scaled, v1_scaled, v0_scaled])
        scaled_vertices.extend([v3_scaled, v2_scaled, v1_scaled])
        
        # Add color for each vertex (4 vertices per tetrahedron)
        for _ in range(12):
            vertex_colors.append(color)
    
    # Convert to numpy arrays
    scaled_vertices = np.array(scaled_vertices)
    vertex_colors   = np.array(vertex_colors)

    print("Scaled tetrahedra vertices:", scaled_vertices.shape)
    
    # Create new tetrahedra indices for scaled vertices
    scaled_tet_indices = np.arange(len(scaled_vertices)).reshape(-1, 3)

    trimesh_obj = trimesh.Trimesh(vertices=scaled_vertices, faces=scaled_tet_indices, vertex_colors=vertex_colors)
    trimesh_obj.show()


def test_manifold():
    """Test Manifold cube creation"""
    print("Testing Manifold...")
    
    # Create a cube
    cube_verts, cube_tris = vocd.create_cube(2.0, 3.0, 4.0)
    
    print(f"Created cube with dimensions 2x3x4")
    print(f"Cube verts: {cube_verts}")
    print(f"Cube tris: {cube_tris}")
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
