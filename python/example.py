#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import vocd
import manifold3d
import trimesh
import numpy as np
from random import random
np.set_printoptions(precision=4, suppress=True)

rand_color = [random(), random(), random()]
def explode(convex_pieces, explode_amount = 1.05, debug_shapes = None):
    global rand_color
    exploded_pieces = []
    for i, convex_piece in enumerate(convex_pieces):
        centroid = np.mean(convex_piece.to_mesh().vert_properties[:, :3], axis=0)
        offset = centroid*explode_amount - centroid
        exploded_piece = convex_piece.translate([offset[0], offset[1], offset[2]])
        rand_color = [random(), random(), random()]
        try:
            exploded_piece  = exploded_piece .set_properties(3, lambda pos, oldProp: rand_color)
            if debug_shapes is not None:
                debug_shapes[i] = debug_shapes[i].set_properties(3, lambda pos, oldProp: rand_color)
                exploded_pieces.append(debug_shapes[i])
        except:
            pass
            #print("Failed to set properties")
        exploded_pieces.append(exploded_piece)
    return exploded_pieces

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
            #print(f"Skipping degenerate tetrahedron with volume {volume:.6f}")
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

def cells_to_manifolds(cells, explode=False):
    # Compute the convex hull of the Voronoi cells
    hulls = []
    for cell in cells:
        points = np.array(cell).reshape(-1, 3)

        if explode:
            # Compute the average of the points to center the hull
            avg_point = np.mean(points, axis=0)
            points -= avg_point  # Center the points around the origin
            points *= 0.8  # Scale down the points
            points += avg_point  # Shift back to original position

        hull = manifold3d.Manifold.hull_points(points)#trimesh.convex.convex_hull(points)
        hulls.append(hull)
    return hulls

def visualize_voronoi(cells):
    """Visualize Voronoi cells using trimesh"""
    hulls = cells_to_manifolds(cells)
    trimesh_hulls = [to_trimesh(hull) for hull in hulls]
    # Create a trimesh scene with all Voronoi cell hulls
    scene = trimesh.Scene(trimesh_hulls)
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
    #print("Vert properties shape:", mesh.vert_properties.shape)
    if mesh.vert_properties.shape[1] > 3:
        vertices =  mesh.vert_properties[:, :3]
        colors   = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
    else:
        vertices = mesh.vert_properties
        colors   = None
    return trimesh.Trimesh(vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors)

#def getCircumCenter(p0, p1, p2, p3):
#    b = p1 - p0
#    c = p2 - p0
#    d = p3 - p0
#
#    det = 2.0 * (b[0]*(c[1]*d[2] - c[2]*d[1]) - b[1]*(c[0]*d[2] - c[2]*d[0]) + b[2]*(c[0]*d[1] - c[1]*d[0]))
#    if det == 0.0:
#        return p0
#    else: 
#        v = np.cross(c, d)*np.dot(b,b) + np.cross(d, b)*np.dot(c,c) + np.cross(b, c)*np.dot(d,d)
#        v /= det
#        return p0 + v

def getCircumCenter(p0, p1, p2, p3):
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        a = np.cross(e2, e3) * np.dot(e1, e1)
        b = np.cross(e3, e1) * np.dot(e2, e2)
        c = np.cross(e1, e2) * np.dot(e3, e3)
        alpha = np.linalg.det([
                    [e1[0], e1[1], e1[2]],
                    [e2[0], e2[1], e2[2]],
                    [e3[0], e3[1], e3[2]]
                ]) * 2.0
        return ((a + b + c) / alpha) + p0

def getTriangleCircumCenter(p0, p1, p2):
        a = p0 - p2
        b = p1 - p2
        c = p0 - p1

        a_length = np.linalg.norm(a)
        b_length = np.linalg.norm(b)
        #c_length = np.linalg.norm(c)

        acrsb = np.cross(a, b)
        numerator = np.cross(b * (a_length * a_length) - a * (b_length * b_length), acrsb)
        crs = np.linalg.norm(acrsb)
        denominator = 2.0 * (crs * crs)

        circumcenter = numerator / denominator + p2
        return circumcenter

def constrain_to_segment(position, a, b):
    ba = b - a
    t = np.dot(position - a, ba) / np.dot(ba, ba)
    return a + t * (b - a) if 0 <= t <= 1 else (a if t < 0 else b)

#def make_non_convex_manifold():
#    cube   = manifold3d.Manifold.cube([1.0, 1.0, 1.0]).translate([-0.5, -0.5, -0.75])
#    sphere = manifold3d.Manifold.sphere(0.7)
#
#    # Create a non-convex manifold by combining a cube and a sphere
#    fun_shape = cube - sphere
#
#    trimesh_obj = to_trimesh(fun_shape)
#    #trimesh_obj.show()
#
#    reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
#
#    tet_vertices, tet_indices = vocd.tetrahedrize(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
#
#    tet_indices  = np.array(tet_indices , dtype=np.uint32)
#    tet_vertices = np.array(tet_vertices, dtype=np.float64)
#
#    # Reduce the tetrahedra to only those that share a face with a reflex face
#    # TODO: THERE IS A FUNNY BUG HERE THAT ONLY FINDS FOUR TETRAHEDRA
#    #       THAT SHARE A FACE WITH THE REFLEX FACES
#    pruned_tet_indices = []
#    for face in reflex_faces:
#        reflex_face_vertices = trimesh_obj.vertices[trimesh_obj.faces[face]]
#        for i in range(tet_indices.shape[0]):
#            this_tet_vertices = tet_vertices[tet_indices[i]]
#
#            # Check if three of the tetrahedron's vertices are within a small distance of the reflex face edges
#            # This is a more robust way to check for shared faces
#            shared_vertices = 0
#            for j in range(3):
#                a = reflex_face_vertices[j]
#                b = reflex_face_vertices[(j+1) % 3]
#                for pair in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]:
#                    # Constrain the tetrahedron vertex to the segment defined by the face edge
#                    constrained_vertex = constrain_to_segment(this_tet_vertices[pair[0]], a, b)
#                    if  < 1e-4:
#                        shared_vertices += 1
#                        break
#
#            #print("Comparing tetrahedron", i, "with reflex face", face)
#
#            #num_shared_vertices = 0
#            #for j in range(3):
#            #    for k in range(4):
#            #        if np.linalg.norm(reflex_face_vertices[j] - this_tet_vertices[k]) < 1e-4:
#            #            num_shared_vertices += 1
#            #            #break
#            #if num_shared_vertices == 3:
#            #    print(f"Tetrahedron {i} shares a face with reflex face {face}")
#            #    pruned_tet_indices.append(tet_indices[i])
#
#    visualize_tetrahedra(tet_vertices, np.array(pruned_tet_indices))
#
#    print(tet_indices.shape, tet_vertices.shape)
#
#    scene_objects = [trimesh_obj, trimesh.Trimesh(vertices=trimesh_obj.vertices, faces=trimesh_obj.faces[reflex_faces], face_colors=np.array([0, 255, 0, 100], dtype=np.uint8))]
#    #for tet in tet_indices:
#    for i in range(len(tet_indices)):
#        p0 = tet_vertices[tet_indices[i, 0]]
#        p1 = tet_vertices[tet_indices[i, 1]]
#        p2 = tet_vertices[tet_indices[i, 2]]
#        p3 = tet_vertices[tet_indices[i, 3]]
#
#        # Check if tetrahedron is degenerate (coplanar)
#        # Calculate volume using scalar triple product
#        volume = np.abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
#        if volume < 0.002:
#            #print(f"Skipping degenerate tetrahedron at index {i} with volume {volume:.6f}")
#            continue
#
#        circum_center = getCircumCenter(p0, p1, p2, p3)
#        #print(f"Tetrahedron: Circumcenter at {circum_center} Radius1 {np.linalg.norm(p0-circum_center)} Radius2 {np.linalg.norm(p1-circum_center)} Radius3 {np.linalg.norm(p2-circum_center)} Radius4 {np.linalg.norm(p3-circum_center)}")
#        scene_objects.append(trimesh.creation.icosphere(subdivisions=1, radius=0.05).apply_translation(circum_center))
#
#    # Create a trimesh scene with the non-convex manifold and tetrahedra circumcenters
#    scene = trimesh.Scene(scene_objects)
#    scene.show()

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

def test_reflex():
    """Test reflex edge detection"""
    print("Testing reflex edge detection...")
    cube  = manifold3d.Manifold.cube([1.0, 1.0, 1.0])
    cube2 = manifold3d.Manifold.cube([1.0, 1.0, 1.0]).translate([-0.25, -0.25, -0.25])
    # Create a non-convex manifold by combining a cube and a sphere
    fun_shape = cube - cube2

    trimesh_obj = to_trimesh(fun_shape)
    #trimesh_obj.show()

    reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    print(f"Reflex faces: {reflex_faces}")

    reflex_edges = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    print(f"Reflex edges: {reflex_edges}")
    print()

def visualize_triangle_convex_decomposition(fun_shape):
    """Visualize convex decomposition using reflex face circumcenters"""
    trimesh_obj = to_trimesh(fun_shape)

    face_set = set()
    reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    if len(reflex_edges_and_faces) > 0:
        for edge_indices, face_indices in reflex_edges_and_faces:
            for face_index in face_indices:
                face_set.add(face_index)
        reflex_faces = np.array(list(face_set), dtype=np.uint32)

        circumcenters = {}
        for face_index in reflex_faces:
            face_vertices = trimesh_obj.vertices[trimesh_obj.faces[face_index]]
            circumcenter = getTriangleCircumCenter(face_vertices[0], face_vertices[1], face_vertices[2])
            #print(f"Face {face_index} circumcenter at {circumcenter}")
            circumcenters[str(circumcenter)] = (circumcenter, np.linalg.norm(circumcenter - face_vertices[0]))

        # Separate into separate arrays
        points = np.array([cc[0] for cc in circumcenters.values()])
        wts    = np.array([cc[1] for cc in circumcenters.values()])
        
        # Define bounds
        bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
        
        # Compute Voronoi cell volumes
        #print(points, wts, bounds)
        cells = vocd.voronoi_3d(points, wts, bounds)
        
        print(f"Computed Voronoi cells for {points.shape} points")
        #print(f"Voronoi cell hull vertices: {cells}")
        print(f"Computed {len(cells)} Voronoi cell volumes")

        hulls = cells_to_manifolds(cells, False)

        intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]

        exploded_hulls = explode(intersected_hulls)
    else:
        exploded_hulls = [fun_shape]

    trimesh_hulls = [to_trimesh(hull) for hull in exploded_hulls]
    scene = trimesh.Scene(trimesh_hulls)
    scene.show()

def visualize_tetrahedron_convex_decomposition(fun_shape):
    """Visualize convex decomposition using reflex face circumcenters"""
    trimesh_obj = to_trimesh(fun_shape)

    tet_vertices, tet_indices = vocd.tetrahedrize(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))

    tet_indices  = np.array(tet_indices , dtype=np.uint32)
    tet_vertices = np.array(tet_vertices, dtype=np.float64)

    circumcenters = {}
    for tetrahedron_indices in tet_indices:
        tetrahedron_vertices = tet_vertices[tetrahedron_indices]
        circumcenter = getCircumCenter(tetrahedron_vertices[0], tetrahedron_vertices[1], tetrahedron_vertices[2], tetrahedron_vertices[3])
        #print(f"Face {face_index} circumcenter at {circumcenter}")
        circumcenters[str(circumcenter)] = (circumcenter, np.linalg.norm(circumcenter - tetrahedron_vertices[0]))

    # Separate into separate arrays
    points = np.array([cc[0] for cc in circumcenters.values()])
    wts    = np.array([cc[1] for cc in circumcenters.values()])
    
    # Define bounds
    bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # Compute Voronoi cell volumes
    #print(points, wts, bounds)
    cells = vocd.voronoi_3d(points, wts, bounds)
    
    print(f"Computed Voronoi cells for {points.shape} points")
    #print(f"Voronoi cell hull vertices: {cells}")
    print(f"Computed {len(cells)} Voronoi cell volumes")

    hulls = cells_to_manifolds(cells, False)

    intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]

    exploded_hulls = explode(intersected_hulls)

    trimesh_hulls = [to_trimesh(hull) for hull in exploded_hulls]
    scene = trimesh.Scene(trimesh_hulls)
    scene.show()

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    #test_tetrahedrization()
    #test_manifold()
    #test_voronoi()
    #make_non_convex_manifold()
    #print(vocd)
    #test_reflex()
    
    #cube  = manifold3d.Manifold.cube([1.0, 1.0, 1.0], True)
    #cube2 = manifold3d.Manifold.sphere(1.0, 64)
    ## Create a non-convex manifold by combining a cube and a sphere
    #fun_shape = (cube - cube2.translate([-0.5, -0.5, -0.5])) - cube2.translate([-0.5, 0.5, -0.5])
    cube  = manifold3d.Manifold.cube([1.0, 1.0, 0.5], True)
    for i in range(5):
        cube = cube - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])
    cube = cube.rotate([0, 180, 90])
    for i in range(5):
        cube = cube - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])

    visualize_triangle_convex_decomposition(cube)

    visualize_tetrahedron_convex_decomposition(cube)
