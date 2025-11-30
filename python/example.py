#!/usr/bin/env python3
"""
Example usage of the vocd module
"""

import vocd_ext as vocd
import manifold3d
import trimesh
import numpy as np
from random import random
import time
import tqdm
import triangle as tr
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import tetrahedralize

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

def dot2(v):
    return np.dot(v, v)

def distance_to_triangle(v1, v2, v3, p ):
    v21 = v2 - v1
    p1  =  p - v1
    v32 = v3 - v2
    p2  =  p - v2
    v13 = v1 - v3
    p3  =  p - v3
    nor = np.cross(v21, v13)
    return np.sqrt( min( min( # 3 edges  
                    dot2(v21*np.clip(np.dot(v21,p1)/dot2(v21), 0.0, 1.0)-p1), 
                    dot2(v32*np.clip(np.dot(v32,p2)/dot2(v32), 0.0, 1.0)-p2) ), 
                    dot2(v13*np.clip(np.dot(v13,p3)/dot2(v13), 0.0, 1.0)-p3) )
                    if (np.sign(np.dot(np.cross(v21,nor),p1)) +  # inside/outside test 
                        np.sign(np.dot(np.cross(v32,nor),p2)) + 
                        np.sign(np.dot(np.cross(v13,nor),p3))<2.0)
                    else np.dot(nor,p1)*np.dot(nor,p1)/dot2(nor) ) # 1 face 

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
    #t0 = time.perf_counter()
    mesh = model.to_mesh()
    #print("Vert properties shape:", mesh.vert_properties.shape)
    if mesh.vert_properties.shape[1] > 3:
        vertices =  mesh.vert_properties[:, :3]
        colors   = (mesh.vert_properties[:, 3:] * 255).astype(np.uint8)
    else:
        vertices = mesh.vert_properties
        colors   = None
    output_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.tri_verts, vertex_colors=colors)
    #print("Converted to trimesh in", time.perf_counter() - t0, "seconds")
    return output_mesh

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

def triangle_circumcenter2(a, b, c):
    ac = c - a
    ab = b - a
    abXac = np.cross(ab, ac)
    #this is the vector from a TO the circumsphere center
    aclen2 = np.dot(ac, ac)
    ablen2 = np.dot(ab, ab)
    abXaclen2 = np.dot(abXac, abXac)
    toCircumsphereCenter = (np.cross(abXac, ab) * aclen2 + np.cross(ac, abXac) * ablen2) / (2.0 * abXaclen2)
    return a  +  toCircumsphereCenter # now this is the actual 3-space location

def triangle_circumcenter3(va, vb, vc):
    a = va - vc
    b = vb - vc
    c = va - vb
    a_length = np.linalg.norm(a)
    b_length = np.linalg.norm(b)
    c_length = np.linalg.norm(c)
    numerator = np.cross((((a_length * a_length) * b) - ((b_length * b_length) * a)), np.cross(a, b))
    crs = np.linalg.norm(np.cross(a, b))
    denominator = 2.0 * (crs * crs)
    circumcenter = (numerator / denominator) + vc
    return circumcenter

def constrain_to_segment(position, a, b):
    ba = b - a
    t = np.dot(position - a, ba) / np.dot(ba, ba)
    return a + t * (b - a) if 0 <= t <= 1 else (a if t < 0 else b)

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
    cells, neighbors = vocd.voronoi_3d(points, wts, bounds)
    
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

def visualize_convex_decomposition(shapes : list[manifold3d.Manifold]):
    global t0
    #t0 = time.perf_counter()

    exploded_hulls = explode(shapes, 1.0)

    trimesh_hulls = [to_trimesh(hull) for hull in exploded_hulls]
    print(f"Convex pieces: {len(shapes)} in {time.perf_counter() - t0:.4f} seconds")
    scene = trimesh.Scene(trimesh_hulls)
    scene.show()

def triangle_convex_decomposition(fun_shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    """Visualize convex decomposition using reflex face circumcenters"""
    global t0
    t0 = time.perf_counter()
    trimesh_obj = to_trimesh(fun_shape)

    face_set = set()
    #reflex_edges, reflex_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    if len(reflex_faces) > 0:
        for face_indices in reflex_faces:
            #for face_index in face_indices:
                face_set.add(face_indices)
        reflex_faces = np.array(list(face_set), dtype=np.uint32)

        circumcenters = {}
        for face_index in reflex_faces:
            face_vertices = trimesh_obj.vertices[trimesh_obj.faces[face_index]]
            circumcenter = triangle_circumcenter3(face_vertices[0], face_vertices[1], face_vertices[2])
            #print(f"Face {face_index} circumcenter at {circumcenter}")
            circumcenters[str(circumcenter)] = (circumcenter, np.linalg.norm(circumcenter - face_vertices[0]))

        # Separate into separate arrays
        points = np.array([cc[0] for cc in circumcenters.values()])
        wts    = np.array([cc[1] for cc in circumcenters.values()])
        
        # Define bounds
        bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
        
        # Compute Voronoi cell volumes
        #print(points, wts, bounds)
        cells, neighbors = vocd.voronoi_3d(points, wts, bounds)
        print(f"Voronoi Cells: {len(cells)} in {time.perf_counter() - t0:.4f} seconds")
        print(f"Neighbors: {neighbors}")
        
        print(f"Computed Voronoi cells for {points.shape} points")
        #print(f"Voronoi cell hull vertices: {cells}")
        print(f"Computed {len(cells)} Voronoi cell volumes")

        hulls = cells_to_manifolds(cells, False)

        intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]
        decomposed_hulls = []
        for hull in intersected_hulls:
            decomposed_hulls += manifold3d.Manifold.decompose(hull)
        recursed_hulls = []
        for i, hull in enumerate(decomposed_hulls):
            trimesh_obj = to_trimesh(hull)
            #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            enforced_hull = manifold3d.Manifold.hull(hull)
            if len(reflex_faces) > 0 and enforced_hull.volume() - hull.volume() > 0.0001:
                print("Recursing on hull", i, "with reflex edges...")
                recursed_hulls += convex_2d_decomposition(hull)#[hull]#
            else:
                recursed_hulls.append(enforced_hull)
        return recursed_hulls
    else:
        return [fun_shape]

def tetrahedron_convex_decomposition(fun_shape) -> list[manifold3d.Manifold]:
    """Visualize convex decomposition using reflex face circumcenters"""
    global t0
    t0 = time.perf_counter()
    trimesh_obj = to_trimesh(fun_shape)

    face_set = set()
    #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    reflex_faces2 = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
    if len(reflex_faces2) > 0:
        for face_indices in reflex_faces2:
            #for face_index in face_indices:
            face_set.add(face_indices)
        reflex_faces = np.array(list(face_set), dtype=np.uint32)
        face_vertices = trimesh_obj.vertices[trimesh_obj.faces[reflex_faces]]

    tet_vertices, tet_indices = vocd.tetrahedrize(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))

    tet_indices  = np.array(tet_indices , dtype=np.uint32)
    tet_vertices = np.array(tet_vertices, dtype=np.float64)

    circumcenters = {}
    for tetrahedron_indices in tet_indices:
        tetrahedron_vertices = tet_vertices[tetrahedron_indices]

        # Keep tetrahedron if it shares a face plane with a reflex face
        face_found = False
        for i in range(4):
            tet_face = np.array([tetrahedron_vertices[j] for j in range(4) if j != i])
            for reflex_face_index in reflex_faces:
                reflex_face = trimesh_obj.vertices[trimesh_obj.faces[reflex_face_index]]
                # Check if the face planes are the same by comparing normals and a point
                tet_normal = np.cross(tet_face[1] - tet_face[0], tet_face[2] - tet_face[0])
                tet_normal /= np.linalg.norm(tet_normal)
                reflex_normal = np.cross(reflex_face[1] - reflex_face[0], reflex_face[2] - reflex_face[0])
                reflex_normal /= np.linalg.norm(reflex_normal)
                if np.allclose(tet_normal, reflex_normal) or np.allclose(tet_normal, -reflex_normal):
                    # Check if a point from the tetrahedron face is close to the reflex face plane
                    d = -np.dot(reflex_normal, reflex_face[0])
                    distance = np.dot(reflex_normal, tet_face[0]) + d
                    if np.abs(distance) < 1e-4:
                        face_found = True
                        break
            if face_found:
                break
        if not face_found:
            continue

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
    cells, neighbors = vocd.voronoi_3d(points, wts, bounds)
    
    print(f"Computed Voronoi cells for {points.shape} points")
    #print(f"Voronoi cell hull vertices: {cells}")
    print(f"Computed {len(cells)} Voronoi cell volumes")

    hulls = cells_to_manifolds(cells, False)

    intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]

    return intersected_hulls

def convex_2d_decomposition(shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    min_vol = 0.00001
    outputs = []
    if shape is None:
        print("[ERROR] SHAPE IS NONE!!!")
        return []
    shapes = shape.decompose()
    if len(shapes) == 0:
        print("[ERROR] INVALID DECOMPOSITION!!!")
        return [shape]
    for shape in shapes:
        if shape is None:
            continue
        cur_trimesh = to_trimesh(shape)

        # A list of the concave start/end segments of the mesh [N, 2, 3]
        concave_segment_indices = cur_trimesh.face_adjacency_edges[~cur_trimesh.face_adjacency_convex]
        concave_segments = cur_trimesh.vertices[concave_segment_indices][:, :, :2]  # Drop Z for 2D processing
        if concave_segment_indices.shape[0] == 0: # Early exit if we're convex already
            outputs += [shape]
            continue

        # Recreate the mapping without duplicates
        points, point_indices = np.unique(concave_segments.reshape(-1, 2), axis=0, return_inverse=True)
        segments = point_indices.reshape(-1, 2)
        points = np.concatenate([points, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])], axis=0)  # Add Outer Triangle
        pslg = dict(vertices=points, segments=segments)

        # Generate a constrained Delaunay triangulation with convex hull edges
        t = tr.triangulate(pslg, 'pc')

        #print(t)
        ## Visualize the triangulation
        #tr.compare(plt, pslg, t)
        #plt.show()

        # For each triangle, extrude to a prism and intersect with the original shape
        for tri_indices in t['triangles']:
            tri_2d = t['vertices'][tri_indices]
            # Create a prism from the triangle
            prism = manifold3d.CrossSection([[(tri_2d[0][0], tri_2d[0][1]),
                                              (tri_2d[1][0], tri_2d[1][1]),
                                              (tri_2d[2][0], tri_2d[2][1])]]).extrude(height=4.0).translate([0.0, 0.0, -2.0])
            convex_shapes = manifold3d.Manifold.decompose(prism ^ shape)
            for convex_shape in convex_shapes:
                #if convex_shape.volume() > min_vol:
                #outputs.append(manifold3d.Manifold.hull(convex_shape))

                trimesh_obj = to_trimesh(convex_shape)
                #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                enforced_hull = manifold3d.Manifold.hull(convex_shape)
                if len(reflex_faces) > 0 and enforced_hull.volume() - convex_shape.volume() > 0.0001:
                    print("Recursing on hull with reflex edges...")
                    #outputs += convex_2d_decomposition(convex_shape)#[hull]#
                    rotated_hull = convex_shape.rotate([90, 0, 0])
                    cur_recursed_hulls = convex_2d_decomposition(rotated_hull)#[hull]#
                    for recursed_hull in cur_recursed_hulls:
                        outputs.append(recursed_hull.rotate([-90, 0, 0]))
                else:
                    outputs.append(enforced_hull)
        #return recursed_hulls

    return outputs

def convex_simple_2d_decomposition(shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    t0 = time.perf_counter()
    min_vol = 0.00001
    outputs = []
    if shape is None:
        print("[ERROR] SHAPE IS NONE!!!")
        return []
    shapes = shape.decompose()
    if len(shapes) == 0:
        print("[ERROR] INVALID DECOMPOSITION!!!")
        return [shape]
    for shape in shapes:
        if shape is None:
            continue
        cur_trimesh = to_trimesh(shape)

        # A list of the concave start/end segments of the mesh [N, 2, 3]
        concave_segment_indices = cur_trimesh.face_adjacency_edges[~cur_trimesh.face_adjacency_convex]
        concave_segments = cur_trimesh.vertices[concave_segment_indices][:, :, :2]  # Drop Z for 2D processing
        if concave_segment_indices.shape[0] == 0: # Early exit if we're convex already
            outputs += [shape]
            continue

        # Recreate the mapping without duplicates
        points, point_indices = np.unique(concave_segments.reshape(-1, 2), axis=0, return_inverse=True)
        points = np.concatenate([points, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])], axis=0)  # Add Outer Triangle
        pslg = dict(vertices=points)

        # Generate a constrained Delaunay triangulation with convex hull edges
        t = tr.triangulate(pslg, 'Dc')

        #print(t)
        # Visualize the triangulation
        #tr.compare(plt, pslg, t)
        #plt.show()

        # For each triangle, extrude to a prism and intersect with the original shape
        for tri_indices in t['triangles']:
            tri_2d = t['vertices'][tri_indices]
            # Create a prism from the triangle
            prism = manifold3d.CrossSection([[(tri_2d[0][0], tri_2d[0][1]),
                                              (tri_2d[1][0], tri_2d[1][1]),
                                              (tri_2d[2][0], tri_2d[2][1])]]).extrude(height=4.0).translate([0.0, 0.0, -2.0])
            convex_shapes = manifold3d.Manifold.decompose(prism ^ shape)
            for convex_shape in convex_shapes:
                #if convex_shape.volume() > min_vol:
                #outputs.append(manifold3d.Manifold.hull(convex_shape))

                trimesh_obj = to_trimesh(convex_shape)
                #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                enforced_hull = manifold3d.Manifold.hull(convex_shape)
                if len(reflex_faces) > 0 and enforced_hull.volume() - convex_shape.volume() > 0.001:
                    print("Recursing on hull with reflex edges...")
                    #outputs += convex_2d_decomposition(convex_shape)#[hull]#
                    rotated_hull = convex_shape.rotate([90, 0, 0])
                    cur_recursed_hulls = convex_simple_2d_decomposition(rotated_hull)#[hull]#
                    for recursed_hull in cur_recursed_hulls:
                        outputs.append(recursed_hull.rotate([-90, 0, 0]))
                else:
                    outputs.append(enforced_hull)
        #return recursed_hulls

    t1 = time.perf_counter()
    print(f"convex_simple_2d_decomposition took {t1 - t0:.4f} seconds")
    return outputs

def get_durable_seg_key(seg):
    return str(np.array(sorted(seg, key=lambda x: (str(x[0]), str(x[1])))))

def convex_simple_manifold_2d_decomposition(shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    t0 = time.perf_counter()
    min_vol = 0.00001
    outputs = []
    if shape is None:
        print("[ERROR] SHAPE IS NONE!!!")
        return []
    shapes = shape.decompose()
    if len(shapes) == 0:
        print("[ERROR] INVALID DECOMPOSITION!!!")
        return [shape]
    for shape in shapes:
        if shape is None:
            continue
        cur_trimesh = to_trimesh(shape)

        # A list of the concave start/end segments of the mesh [N, 2, 3]
        concave_segment_indices = cur_trimesh.face_adjacency_edges[~cur_trimesh.face_adjacency_convex]
        concave_segments = cur_trimesh.vertices[concave_segment_indices][:, :, :2]  # Drop Z for 2D processing
        if concave_segment_indices.shape[0] == 0: # Early exit if we're convex already
            outputs += [shape]
            continue

        print(concave_segments.tolist())

        edge_set = set()
        for seg in concave_segments:
            edge_set.add(get_durable_seg_key(seg))

        # Remove duplicates
        points, point_indices = np.unique(concave_segments.reshape(-1, 2), axis=0, return_inverse=True)
        points_2d = np.concatenate([points, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])], axis=0)  # Add Bounding Square
    
        # Compute the lifted weighted points and delaunay triangulation via Convex Hull
        S_norm = np.sum(points_2d ** 2, axis = 1) - 1.0 ** 2 # Swap 1.0 for point radius if desired
        S_lifted = np.concatenate([points_2d, S_norm[:,None]], axis = 1)
        convex_hull = manifold3d.Manifold.hull_points(S_lifted)
        convex_trimesh = to_trimesh(convex_hull)

        # Accumulate triangles with the correct winding order
        triangles = []
        verts_2d = convex_trimesh.vertices[:,:2]
        for tri_indices in tqdm.tqdm(convex_trimesh.faces, desc="Processing Triangles"):
            tri_2d = verts_2d[tri_indices]
            # Check winding order using cross product
            v0      = tri_2d[1] - tri_2d[0]
            v1      = tri_2d[2] - tri_2d[0]
            cross_z = v0[0] * v1[1] - v0[1] * v1[0]
            if cross_z < 0:
                # Keep only triangles that share an edge with the concave segments
                shared_edge = True
                for i in range(3):
                    tri_edge = np.array([tri_2d[i], tri_2d[(i + 1) % 3]])
                    if edge_set.__contains__(get_durable_seg_key(tri_edge)):
                        shared_edge = True
                        break
                if shared_edge:
                    triangles.append([tri_indices[0], tri_indices[1], tri_indices[2]])
        triangles = np.array(triangles, dtype=np.int32)

        # Plot triangles in matplotlib
        print("Triangles Shape", triangles.shape)
        #plt.triplot(convex_trimesh.vertices[:,0], convex_trimesh.vertices[:,1], triangles)
        #plt.show()

        use_voronoi = False

        if not use_voronoi:
            recursed_hulls = []
            # For each triangle, extrude to a prism and intersect with the original shape
            for tri_indices in triangles:
                tri_2d = verts_2d[tri_indices]
                # Create a prism from the triangle
                prism = manifold3d.CrossSection([[(tri_2d[0][0], tri_2d[0][1]),
                                                  (tri_2d[2][0], tri_2d[2][1]),
                                                  (tri_2d[1][0], tri_2d[1][1])]]).extrude(height=4.0).translate([0.0, 0.0, -2.0])
                decomposed_hulls = manifold3d.Manifold.decompose(prism ^ shape)
                for i, hull in enumerate(decomposed_hulls):
                    #trimesh_obj = to_trimesh(hull)
                    #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                    #reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                    #enforced_hull = manifold3d.Manifold.hull(hull)
                    #if len(reflex_faces) > 0 and enforced_hull.volume() - hull.volume() > 0.0000001:
                    #    print("Recursing on hull", i, "with reflex edges...")
                    #    rotated_hull = hull.rotate([90, 0, 0])
                    #    cur_recursed_hulls = convex_simple_manifold_2d_decomposition(rotated_hull)#[hull]#
                    #    for recursed_hull in cur_recursed_hulls:
                    #        recursed_hulls.append(recursed_hull.rotate([-90, 0, 0]))
                    #else:
                    recursed_hulls.append(hull)
            return recursed_hulls
        else:
            circumcenters =[]
            circumradii = []
            for tri_indices in triangles:
                tri_2d = verts_2d[tri_indices]
                #print("Before", tri_2d)
                tri_2d = np.array([[tri_2d[0][0], tri_2d[0][1], 0.0],
                                [tri_2d[1][0], tri_2d[1][1], 0.0],
                                [tri_2d[2][0], tri_2d[2][1], 0.0]]) # Add Z=0 for 3D processing
                circumcenters.append(triangle_circumcenter3(tri_2d[0], tri_2d[1], tri_2d[2]))
                circumradii.append(np.linalg.norm(circumcenters[-1] - tri_2d[0]))
            
            circumcenters = np.array(circumcenters)
            circumradii   = np.array(circumradii)

            #plt.scatter(circumcenters[:, 0], circumcenters[:, 1], s=circumradii * 100)
            #plt.show()

            # Define bounds
            bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
            
            # Compute Voronoi cell volumes
            #print(circumcenters, circumradii, bounds)
            cells, neighbors = vocd.voronoi_3d(circumcenters, circumradii, bounds)
            print(f"Voronoi Cells: {len(cells)}")

            hulls = cells_to_manifolds(cells, False)

            intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]
            decomposed_hulls = []
            for hull in intersected_hulls:
                decomposed_hulls += manifold3d.Manifold.decompose(hull)
            recursed_hulls = []
            for i, hull in enumerate(decomposed_hulls):
                trimesh_obj = to_trimesh(hull)
                #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
                enforced_hull = manifold3d.Manifold.hull(hull)
                #if len(reflex_faces) > 0 and enforced_hull.volume() - hull.volume() > 0.0000001:
                #    print("Recursing on hull", i, "with reflex edges...")
                #    rotated_hull = hull.rotate([90, 0, 0])
                #    cur_recursed_hulls = convex_simple_manifold_2d_decomposition(rotated_hull)#[hull]#
                #    for recursed_hull in cur_recursed_hulls:
                #        recursed_hulls.append(recursed_hull.rotate([-90, 0, 0]))
                #else:
                recursed_hulls.append(enforced_hull)
            return recursed_hulls
    else:
        return [fun_shape]

def voronoi_convex_2d_decomposition(shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    min_vol = 0.00001
    outputs = []
    if shape is None:
        print("[ERROR] SHAPE IS NONE!!!")
        return []
    shapes = shape.decompose()
    if len(shapes) == 0:
        print("[ERROR] INVALID DECOMPOSITION!!!")
        return [shape]
    for shape in shapes:
        if shape is None:
            continue
        cur_trimesh = to_trimesh(shape)

        # A list of the concave start/end segments of the mesh [N, 2, 3]
        concave_segment_indices = cur_trimesh.face_adjacency_edges[~cur_trimesh.face_adjacency_convex]
        concave_segments = cur_trimesh.vertices[concave_segment_indices][:, :, :2]  # Drop Z for 2D processing
        if concave_segment_indices.shape[0] == 0: # Early exit if we're convex already
            outputs += [shape]
            continue

        # Recreate the mapping without duplicates
        points, point_indices = np.unique(concave_segments.reshape(-1, 2), axis=0, return_inverse=True)
        segments = point_indices.reshape(-1, 2)
        points = np.concatenate([points, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])], axis=0)  # Add Outer Triangle
        pslg = dict(vertices=points, segments=segments)

        # Generate a constrained Delaunay triangulation with convex hull edges
        t = tr.triangulate(pslg, 'cD')

        #print(t)
        # Visualize the triangulation
        #tr.compare(plt, pslg, t)
        #plt.show()

        # For each triangle, extrude to a prism and intersect with the original shape
        circumcenters =[]
        circumradii = []
        for tri_indices in t['triangles']:
            tri_2d = t['vertices'][tri_indices]
            #print("Before", tri_2d)
            tri_2d = np.array([[tri_2d[0][0], tri_2d[0][1], 0.0],
                               [tri_2d[1][0], tri_2d[1][1], 0.0],
                               [tri_2d[2][0], tri_2d[2][1], 0.0]]) # Add Z=0 for 3D processing
            #print("After", tri_2d)

            # TODO: Only take triangles with at least one edge on a concave segment

            circumcenters.append(triangle_circumcenter3(tri_2d[0], tri_2d[1], tri_2d[2]))
            circumradii.append(np.linalg.norm(circumcenters[-1] - tri_2d[0]))
        
        circumcenters = np.array(circumcenters)
        circumradii   = np.array(circumradii)

        #plt.scatter(circumcenters[:, 0], circumcenters[:, 1], s=circumradii * 100)
        #plt.show()

        # Define bounds
        bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
        
        # Compute Voronoi cell volumes
        #print(circumcenters, circumradii, bounds)
        cells, neighbors = vocd.voronoi_3d(circumcenters, circumradii, bounds)
        print(f"Voronoi Cells: {len(cells)}")

        hulls = cells_to_manifolds(cells, False)

        intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]
        decomposed_hulls = []
        for hull in intersected_hulls:
            decomposed_hulls += manifold3d.Manifold.decompose(hull)
        recursed_hulls = []
        for i, hull in enumerate(decomposed_hulls):
            trimesh_obj = to_trimesh(hull)
            #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            enforced_hull = manifold3d.Manifold.hull(hull)
            if len(reflex_faces) > 0 and enforced_hull.volume() - hull.volume() > 0.0000001:
                print("Recursing on hull", i, "with reflex edges...")
                rotated_hull = hull.rotate([90, 0, 0])
                cur_recursed_hulls = convex_2d_decomposition(rotated_hull)#[hull]#
                for recursed_hull in cur_recursed_hulls:
                    recursed_hulls.append(recursed_hull.rotate([-90, 0, 0]))
            else:
                recursed_hulls.append(enforced_hull)
        return recursed_hulls
    else:
        return [fun_shape]

def get_durable_face_key(face):
    #avg = np.mean(face, axis=0)
    return str(np.array(sorted(face, key=lambda x: (str(x[0]), str(x[1]), str(x[2])))))

def mmf_tetrahedron_convex_decomposition(fun_shape : manifold3d.Manifold) -> list[manifold3d.Manifold]:
    """Visualize convex decomposition using reflex face circumcenters"""
    global t0
    t0 = time.perf_counter()

    min_vol = 0.00001
    outputs = []
    if fun_shape is None:
        print("[ERROR] SHAPE IS NONE!!!")
        return []
    shapes = fun_shape.decompose()
    if len(shapes) == 0:
        print("[ERROR] INVALID DECOMPOSITION!!!")
        return [fun_shape]
    for shape in shapes:
        if shape is None:
            continue

        trimesh_obj = to_trimesh(shape)

        face_set = set()
        durable_reflex_face_keys = set()
        #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
        reflex_faces2 = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
        if len(reflex_faces2) > 0:
            for face_indices in reflex_faces2:
                #for face_index in face_indices:
                face_set.add(face_indices)
            reflex_faces = np.array(list(face_set), dtype=np.uint32)
            face_vertices = trimesh_obj.vertices[trimesh_obj.faces[reflex_faces]]

            for face_index in reflex_faces:
                face = trimesh_obj.vertices[trimesh_obj.faces[face_index]]
                durable_reflex_face_keys.add(get_durable_face_key(face))

        print(face_vertices.shape)

        # Deduplicate face vertices
        unique_face_vertices = np.unique(face_vertices.reshape(-1, 3), axis=0)

        ## Add cube corners based on min/max bounds to improve tetrahedralization
        min_bounds = np.min(unique_face_vertices, axis=0) - 0.1
        max_bounds = np.max(unique_face_vertices, axis=0) + 0.1
        cube_corners = np.array([[min_bounds[0], min_bounds[1], min_bounds[2]],
                                [max_bounds[0], min_bounds[1], min_bounds[2]],
                                [min_bounds[0], max_bounds[1], min_bounds[2]],
                                [max_bounds[0], max_bounds[1], min_bounds[2]],
                                [min_bounds[0], min_bounds[1], max_bounds[2]],
                                [max_bounds[0], min_bounds[1], max_bounds[2]],
                                [min_bounds[0], max_bounds[1], max_bounds[2]],
                                [max_bounds[0], max_bounds[1], max_bounds[2]]], dtype=np.float64)
        unique_face_vertices = np.vstack([unique_face_vertices, cube_corners])
        tet_indices = tetrahedralize.tetrahedralize(unique_face_vertices, min_quality=0.0)#0.005)

        tet_indices  = np.array(tet_indices , dtype=np.uint32)
        tet_vertices = np.array(unique_face_vertices, dtype=np.float64)

        circumcenters = {}
        for tetrahedron_indices in tqdm.tqdm(tet_indices, desc="Pruning non-reflex tetrahedra"):
            tetrahedron_vertices = tet_vertices[tetrahedron_indices]

            # Keep tetrahedron if it shares a face  with a reflex face
            face_found = False
            for i in range(4):
                tet_face = np.array([tetrahedron_vertices[j] for j in range(4) if j != i])
                if durable_reflex_face_keys.__contains__(get_durable_face_key(tet_face)):
                    face_found = True
                    break
            if not face_found:
                continue

            circumcenter = getCircumCenter(tetrahedron_vertices[0], tetrahedron_vertices[1], tetrahedron_vertices[2], tetrahedron_vertices[3])
            #print(f"Face {face_index} circumcenter at {circumcenter}")
            circumcenters[str(circumcenter)] = (circumcenter, np.linalg.norm(circumcenter - tetrahedron_vertices[0]))

        if len(circumcenters) == 0:
            print("No circumcenters found, returning hull of shape")
            return [fun_shape]

        # Separate into separate arrays
        points = np.array([cc[0] for cc in circumcenters.values()])
        wts    = np.array([cc[1] for cc in circumcenters.values()])
        
        # Define bounds
        bounds = np.array([min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2]], dtype=np.float64)  # [x_min, x_max, y_min, y_max, z_min, z_max]
        
        print(f"Computing Voronoi for {points.shape} points...")

        # Compute Voronoi cell volumes
        #print(points, wts, bounds)
        cells, neighbors = vocd.voronoi_3d(points, wts, bounds)
        
        print(f"Computed Voronoi cells for {points.shape} points")
        #print(f"Voronoi cell hull vertices: {cells}")
        print(f"Computed {len(cells)} Voronoi cell volumes")

        hulls = cells_to_manifolds(cells, False)

        intersected_hulls = [fun_shape ^ hull for hull in hulls if not (fun_shape ^ hull).is_empty()]

        for i in range(3):
            decomposed_hulls = []
            for hull in intersected_hulls:
                decomposed_hulls += manifold3d.Manifold.decompose(hull)

            intersected_hulls = []
            intersected_hulls = [hull.as_original().simplify(0.00000001) for hull in decomposed_hulls if not hull.is_empty()]

        decomposed_hulls = []
        for hull in intersected_hulls:
            decomposed_hulls += manifold3d.Manifold.decompose(hull)

        intersected_hulls = []
        intersected_hulls = [hull for hull in decomposed_hulls if not hull.is_empty()]#.volume() > 0.000001]

        print(f"Decomposed into {len(intersected_hulls)} hulls after intersection and decomposition")
        outputs += intersected_hulls

        #for i, hull5 in enumerate(intersected_hulls):
        #    trimesh_obj2 = to_trimesh(hull5)
        #    #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
        #    reflex_faces4 = get_reflex_faces(np.array(trimesh_obj2.vertices), np.array(trimesh_obj2.faces))
        #    enforced_hull = manifold3d.Manifold.hull(hull5)
        #    if len(reflex_faces4) > 0 and enforced_hull.volume() - hull5.volume() > 0.0001:
        #        print("Recursing on hull", i, "with reflex edges...")
        #        rotated_hull = hull5.rotate([90, 0, 0])
        #        cur_recursed_hulls = mmf_tetrahedron_convex_decomposition(rotated_hull)#]
        #        for recursed_hull in cur_recursed_hulls:
        #            outputs.append(recursed_hull.rotate([-90, 0, 0]))
        #    else:
        #        outputs.append(manifold3d.Manifold.hull(hull5))
    return outputs

#def CGAL_convex_decomposition(fun_shape) -> list[manifold3d.Manifold]:
#    """Visualize convex decomposition using CGAL"""
#    trimesh_obj = to_trimesh(fun_shape)
#    return [manifold3d.Manifold.hull_points(np.array(vertices).reshape(-1, 3)) for vertices in 
#             vocd.cgal_convex_decompose_mesh(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))]

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    #cube  = manifold3d.Manifold.cube([1.0, 1.0, 1.0], True)
    #cube2 = manifold3d.Manifold.sphere(1.0, 32)
    #fun_shape = (cube - cube2.translate([-0.5, -0.5, -0.5])) - cube2.translate([-0.5, 0.5, -0.5])
    #fun_shape  = manifold3d.Manifold.cube([1.0, 1.0, 0.5], True)
    #for i in range(5):
    #    fun_shape = fun_shape - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])
    #fun_shape = fun_shape.rotate([0, 180, 90])
    #for i in range(5):
    #    fun_shape = fun_shape - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])

    sphere = manifold3d.Manifold.sphere(0.6, 16)
    cube   = manifold3d.Manifold.cube  ([1.0, 1.0, 1.0], True)
    fun_shape = cube - sphere
    #fun_shape = fun_shape.rotate([15, 15, 15])

    #fun_shape = manifold3d.Manifold.sphere(0.6, 16) + manifold3d.Manifold.sphere(0.6, 16).translate([0.3, 0.3, 0.3])

    #visualize_convex_decomposition(triangle_convex_decomposition(fun_shape))

    #visualize_convex_decomposition(tetrahedron_convex_decomposition(fun_shape))
    visualize_convex_decomposition(mmf_tetrahedron_convex_decomposition(fun_shape))

    ##fun_shape = fun_shape.rotate([0, 90, 0])

    #visualize_convex_decomposition(CGAL_convex_decomposition(fun_shape))

    #visualize_convex_decomposition(convex_2d_decomposition(fun_shape))
    #visualize_convex_decomposition(convex_simple_2d_decomposition(fun_shape))
    #visualize_convex_decomposition(convex_simple_manifold_2d_decomposition(fun_shape))
    #visualize_convex_decomposition(voronoi_convex_2d_decomposition(fun_shape))

    # Reddit Convex Decomposition
    # Have triangles incrementally add their non-reflex neighbors to a convex hull
    # Check to see if the hull remains inside of the original mesh. If so, continue adding.
    # For the inside check, check if the vertex is shared with the original mesh or do a raycast test (use BVH)

    # Use adapted reddit method to consolidate tetrahedra or cells into larger convex pieces, using similar insidedness test.
    # Consolidation could be done as a post-process after either triangle-based, tetrahedron/cell-based, or CGAL convex decomposition.
