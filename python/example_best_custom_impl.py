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

def dot2(v):
    return np.dot(v, v)

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

def visualize_convex_decomposition(shapes : list[manifold3d.Manifold]):
    global t0
    #t0 = time.perf_counter()

    exploded_hulls = explode(shapes)

    trimesh_hulls = [to_trimesh(hull) for hull in exploded_hulls]
    print(f"Convex pieces: {len(shapes)} in {time.perf_counter() - t0:.4f} seconds")
    scene = trimesh.Scene(trimesh_hulls)
    scene.show()

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

        intersected_hulls = [fun_shape ^ hull for hull in hulls]

        intersected_hulls = [hull for hull in intersected_hulls if not hull.is_empty()]
        for i in range(3):
            decomposed_hulls = []
            for hull in intersected_hulls:
                decomposed_hulls += manifold3d.Manifold.decompose(hull)

            intersected_hulls = [hull.as_original().simplify(0.0000001) for hull in decomposed_hulls]

        decomposed_hulls = []
        for hull in intersected_hulls:
            decomposed_hulls += manifold3d.Manifold.decompose(hull)

        intersected_hulls = [hull for hull in decomposed_hulls if not hull.is_empty()]#.volume() > 0.000001]

        print(f"Decomposed into {len(intersected_hulls)} hulls after intersection and decomposition")
        #return intersected_hulls

        recursed_hulls = []
        for i, hull in enumerate(intersected_hulls):
            trimesh_obj = to_trimesh(hull)
            #reflex_edges_and_faces = vocd.getReflexEdges(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            reflex_faces = get_reflex_faces(np.array(trimesh_obj.vertices), np.array(trimesh_obj.faces))
            enforced_hull = manifold3d.Manifold.hull(hull)
            if len(reflex_faces) > 0 and enforced_hull.volume() - hull.volume() > 0.0001:
                print("Recursing on hull", i, "with reflex edges...")
                rotated_hull = hull.rotate([90, 0, 0])
                cur_recursed_hulls = mmf_tetrahedron_convex_decomposition(rotated_hull)#]
                for recursed_hull in cur_recursed_hulls:
                    recursed_hulls.append(recursed_hull.rotate([-90, 0, 0]))
            else:
                recursed_hulls.append(hull)

        recursed_hulls = [manifold3d.Manifold.hull(hull) for hull in recursed_hulls]
        return recursed_hulls

if __name__ == "__main__":
    print(f"Geometry Tools version: {vocd.__version__}")
    print()
    
    #cube  = manifold3d.Manifold.cube([1.0, 1.0, 1.0], True)
    #cube2 = manifold3d.Manifold.sphere(1.0, 32)
    ## Create a non-convex manifold by combining a cube and a sphere
    #fun_shape = (cube - cube2.translate([-0.5, -0.5, -0.5])) - cube2.translate([-0.5, 0.5, -0.5])
    #fun_shape  = manifold3d.Manifold.cube([1.0, 1.0, 0.5], True)
    #for i in range(5):
    #    fun_shape = fun_shape - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])
    #fun_shape = fun_shape.rotate([0, 180, 90])
    #for i in range(5):
    #    fun_shape = fun_shape - manifold3d.Manifold.cube([1.2, 0.5, 0.5], True).rotate([45, 0, 0]).translate([0, i * 0.15 - 0.25, -0.55])

    sphere = manifold3d.Manifold.sphere(0.6, 32)
    cube   = manifold3d.Manifold.cube  ([1.0, 1.0, 1.0], True)
    fun_shape = cube - sphere
    #fun_shape = fun_shape.rotate([15, 15, 15])

    #fun_shape = manifold3d.Manifold.sphere(0.6, 16) + manifold3d.Manifold.sphere(0.6, 16).translate([0.3, 0.3, 0.3])

    visualize_convex_decomposition(mmf_tetrahedron_convex_decomposition(fun_shape))

