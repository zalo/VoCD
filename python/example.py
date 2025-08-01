#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import vocd
import manifold3d
import trimesh
import numpy as np

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
            print(f"Skipping degenerate tetrahedron with volume {volume:.6f}")
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

def visualize_voronoi(cells):
    """Visualize Voronoi cells using trimesh"""
    # Compute the convex hull of the Voronoi cells
    hulls = []
    for cell in cells:
        points = np.array(cell).reshape(-1, 3)

        # Compute the average of the points to center the hull
        avg_point = np.mean(points, axis=0)
        points -= avg_point  # Center the points around the origin
        points *= 0.8  # Scale down the points
        points += avg_point  # Shift back to original position

        hull = trimesh.convex.convex_hull(points)
        hulls.append(hull)
    
    # Create a trimesh scene with all Voronoi cell hulls
    scene = trimesh.Scene(hulls)
    scene.show()

def visualize_mesh(cube_verts, cube_tris):
    """Visualize a mesh using trimesh"""
    cube_mesh = trimesh.Trimesh(vertices=cube_verts, faces=cube_tris)
    cube_mesh.show()

def get_reflex_faces(verts, tris):
    """Get reflex faces from a mesh"""
    # Convert to trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=tris)
    # Get a list of all the non-convex facepairs
    reflex_face_pairs = mesh.face_adjacency[mesh.face_adjacency_convex == False]
    # Create a list of reflex faces
    reflex_face_set = set()
    for pair in reflex_face_pairs:
        reflex_face_set.add(pair[0])
        reflex_face_set.add(pair[1])
    # Return indices of reflex faces
    return np.array(list(reflex_face_set), dtype=np.uint32)

def to_trimesh(model):
    mesh = model.to_mesh()
    print("Vert properties shape:", mesh.vert_properties.shape)
    if mesh.vert_properties.shape[1] > 3:
        vertices =  mesh.vert_properties[:, :3]
        colors   = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
    else:
        vertices = mesh.vert_properties
        colors   = None
    return trimesh.Trimesh(vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors)

def getCircumCenter(p0, p1, p2, p3):
    b = p1 - p0
    c = p2 - p0
    d = p3 - p0

    det = 2.0 * (b[0]*(c[1]*d[2] - c[2]*d[1]) - b[1]*(c[0]*d[2] - c[2]*d[0]) + b[2]*(c[0]*d[1] - c[1]*d[0]))
    if det == 0.0:
        return p0
    else: 
        v = np.cross(c, d)*np.dot(b,b) + np.cross(d, b)*np.dot(c,c) + np.cross(b, c)*np.dot(d,d)
        v /= det
        return p0 + v

def make_non_convex_manifold():
    cube = manifold3d.Manifold.cube([1.0, 1.0, 1.0]).translate([-0.5, -0.5, -0.75])
    sphere = manifold3d.Manifold.sphere(0.7)

    # Create a non-convex manifold by combining a cube and a sphere
    fun_shape = cube - sphere

    trimesh_obj = to_trimesh(fun_shape)
    #trimesh_obj.show()

    reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))

    tet_vertices, tet_indices = vocd.tetrahedrize(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))

    # TODO: Reduce the tetrahedra to only those that share a face with a reflex face
    reflex_faces_mesh = trimesh.Trimesh(vertices=trimesh_obj.vertices, faces=trimesh_obj.faces[reflex_faces], face_colors=np.array([0, 255, 0, 100], dtype=np.uint8))

    visualize_tetrahedra(tet_vertices, tet_indices)

    tet_indices  = np.array(tet_indices, dtype=np.uint32)
    tet_vertices = np.array(tet_vertices, dtype=np.float64)

    print(tet_indices.shape, tet_vertices.shape)

    scene_objects = [trimesh_obj, reflex_faces_mesh]
    #for tet in tet_indices:
    for i in range(len(tet_indices)):
        p0 = tet_vertices[tet_indices[i, 0]]
        p1 = tet_vertices[tet_indices[i, 1]]
        p2 = tet_vertices[tet_indices[i, 2]]
        p3 = tet_vertices[tet_indices[i, 3]]

        # Check if tetrahedron is degenerate (coplanar)
        # Calculate volume using scalar triple product
        volume = np.abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
        if volume < 0.002:
            print(f"Skipping degenerate tetrahedron at index {i} with volume {volume:.6f}")
            continue

        circum_center = getCircumCenter(p0, p1, p2, p3)
        print(f"Tetrahedron: Circumcenter at {circum_center} Radius1 {np.linalg.norm(p0-circum_center)} Radius2 {np.linalg.norm(p1-circum_center)} Radius3 {np.linalg.norm(p2-circum_center)} Radius4 {np.linalg.norm(p3-circum_center)}")
        scene_objects.append(trimesh.creation.icosphere(subdivisions=1, radius=0.05).apply_translation(circum_center))

    # Create a trimesh scene with the non-convex manifold and tetrahedra circumcenters
    scene = trimesh.Scene(scene_objects)
    scene.show()

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

    visualize_voronoi(cells)

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    #test_tetrahedrization()
    #test_manifold()
    #test_voronoi()
    make_non_convex_manifold()
