# Delaunay Tetrahedralization
# Based on code by Matthias Mueller - Ten Minute Physics
# https://www.youtube.com/channel/UCTG_vrRdKYfrpqCv_WV4eyA
# www.matthiasMueller.info/tenMinutePhysics
#
# Original License: MIT

import numpy as np
from functools import cmp_to_key
from random import random
import tqdm

# Face indices for each face of a tetrahedron
TET_FACES = [[2, 1, 0], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


def get_circum_center(p0, p1, p2, p3):
    """Compute the circumcenter of a tetrahedron."""
    b = p1 - p0
    c = p2 - p0
    d = p3 - p0

    det = 2.0 * (b[0] * (c[1] * d[2] - c[2] * d[1]) -
                 b[1] * (c[0] * d[2] - c[2] * d[0]) +
                 b[2] * (c[0] * d[1] - c[1] * d[0]))

    if det == 0.0:
        return p0.copy()

    b_dot = np.dot(b, b)
    c_dot = np.dot(c, c)
    d_dot = np.dot(d, d)

    v = (np.cross(c, d) * b_dot +
         np.cross(d, b) * c_dot +
         np.cross(b, c) * d_dot)
    v /= det
    return p0 + v


def tet_quality(p0, p1, p2, p3):
    """
    Compute tetrahedron quality metric.
    Returns 1.0 for a regular tetrahedron.
    """
    d0 = p1 - p0
    d1 = p2 - p0
    d2 = p3 - p0
    d3 = p2 - p1
    d4 = p3 - p2
    d5 = p1 - p3

    s0 = np.linalg.norm(d0)
    s1 = np.linalg.norm(d1)
    s2 = np.linalg.norm(d2)
    s3 = np.linalg.norm(d3)
    s4 = np.linalg.norm(d4)
    s5 = np.linalg.norm(d5)

    ms = (s0*s0 + s1*s1 + s2*s2 + s3*s3 + s4*s4 + s5*s5) / 6.0
    rms = np.sqrt(ms)

    s = 12.0 / np.sqrt(2.0)
    vol = np.dot(d0, np.cross(d1, d2)) / 6.0

    if rms == 0:
        return 0.0
    return s * vol / (rms * rms * rms)


def _compare_edges(e0, e1):
    """Compare two edges for sorting."""
    if e0[0] < e1[0] or (e0[0] == e1[0] and e0[1] < e1[1]):
        return -1
    return 1


def _equal_edges(e0, e1):
    """Check if two edges are equal."""
    return e0[0] == e1[0] and e0[1] == e1[1]


def _rand_eps():
    """Generate a small random perturbation to avoid degeneracies."""
    eps = 0.0001
    return -eps + 2.0 * random() * eps


def _create_tet_ids(verts, min_quality):
    """
    Core Delaunay tetrahedralization algorithm.

    Args:
        verts: List of numpy arrays (3D points), with the last 4 being the big tet
        min_quality: Minimum tetrahedron quality threshold

    Returns:
        List of tetrahedron vertex indices (flat, 4 indices per tet)
    """
    tet_ids = []
    neighbors = []
    tet_marks = []
    tet_mark = 0
    first_free_tet = -1

    planes_n = []
    planes_d = []

    first_big = len(verts) - 4

    # First big tet
    tet_ids.extend([first_big, first_big + 1, first_big + 2, first_big + 3])
    tet_marks.append(0)

    for i in range(4):
        neighbors.append(-1)
        p0 = verts[first_big + TET_FACES[i][0]]
        p1 = verts[first_big + TET_FACES[i][1]]
        p2 = verts[first_big + TET_FACES[i][2]]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm > 0:
            n = n / norm
        planes_n.append(n)
        planes_d.append(np.dot(p0, n))

    center = np.zeros(3)

    for i in tqdm.tqdm(range(first_big), desc="Inserting vertices into tetrahedralization"):
        p = verts[i]

        # Find non-deleted tet
        tet_nr = 0
        while tet_ids[4 * tet_nr] < 0:
            tet_nr += 1

        # Find containing tet
        tet_mark += 1
        found = False

        while not found:
            if tet_nr < 0 or tet_marks[tet_nr] == tet_mark:
                break
            tet_marks[tet_nr] = tet_mark

            id0 = tet_ids[4 * tet_nr]
            id1 = tet_ids[4 * tet_nr + 1]
            id2 = tet_ids[4 * tet_nr + 2]
            id3 = tet_ids[4 * tet_nr + 3]

            center = (verts[id0] + verts[id1] + verts[id2] + verts[id3]) * 0.25

            min_t = float('inf')
            min_face_nr = -1

            for j in range(4):
                n = planes_n[4 * tet_nr + j]
                d = planes_d[4 * tet_nr + j]

                hp = np.dot(n, p) - d
                hc = np.dot(n, center) - d

                t = hp - hc
                if t == 0:
                    continue

                t = -hc / t

                if 0.0 <= t < min_t:
                    min_t = t
                    min_face_nr = j

            if min_t >= 1.0:
                found = True
            else:
                tet_nr = neighbors[4 * tet_nr + min_face_nr]

        if not found:
            print("*********** failed to insert vertex")
            continue

        # Find violating tets
        tet_mark += 1
        violating_tets = []
        stack = [tet_nr]

        while stack:
            tet_nr = stack.pop()
            if tet_marks[tet_nr] == tet_mark:
                continue
            tet_marks[tet_nr] = tet_mark
            violating_tets.append(tet_nr)

            for j in range(4):
                n = neighbors[4 * tet_nr + j]
                if n < 0 or tet_marks[n] == tet_mark:
                    continue

                # Delaunay condition test
                id0 = tet_ids[4 * n]
                id1 = tet_ids[4 * n + 1]
                id2 = tet_ids[4 * n + 2]
                id3 = tet_ids[4 * n + 3]

                c = get_circum_center(verts[id0], verts[id1], verts[id2], verts[id3])
                r = np.linalg.norm(verts[id0] - c)
                if np.linalg.norm(p - c) < r:
                    stack.append(n)

        # Remove old tets, create new ones
        edges = []

        for j in range(len(violating_tets)):
            tet_nr = violating_tets[j]

            # Copy info before deletion
            ids = [tet_ids[4 * tet_nr + k] for k in range(4)]
            ns = [neighbors[4 * tet_nr + k] for k in range(4)]

            # Delete the tet
            tet_ids[4 * tet_nr] = -1
            tet_ids[4 * tet_nr + 1] = first_free_tet
            first_free_tet = tet_nr

            # Visit neighbors
            for k in range(4):
                n = ns[k]
                if n >= 0 and tet_marks[n] == tet_mark:
                    continue

                # Create new tet
                new_tet_nr = first_free_tet

                if new_tet_nr >= 0:
                    first_free_tet = tet_ids[4 * first_free_tet + 1]
                else:
                    new_tet_nr = len(tet_ids) // 4
                    tet_marks.append(0)
                    for _ in range(4):
                        tet_ids.append(-1)
                        neighbors.append(-1)
                        planes_n.append(np.zeros(3))
                        planes_d.append(0.0)

                id0 = ids[TET_FACES[k][2]]
                id1 = ids[TET_FACES[k][1]]
                id2 = ids[TET_FACES[k][0]]

                tet_ids[4 * new_tet_nr] = id0
                tet_ids[4 * new_tet_nr + 1] = id1
                tet_ids[4 * new_tet_nr + 2] = id2
                tet_ids[4 * new_tet_nr + 3] = i

                neighbors[4 * new_tet_nr] = n

                if n >= 0:
                    for l in range(4):
                        if neighbors[4 * n + l] == tet_nr:
                            neighbors[4 * n + l] = new_tet_nr

                neighbors[4 * new_tet_nr + 1] = -1
                neighbors[4 * new_tet_nr + 2] = -1
                neighbors[4 * new_tet_nr + 3] = -1

                for l in range(4):
                    tp0 = verts[tet_ids[4 * new_tet_nr + TET_FACES[l][0]]]
                    tp1 = verts[tet_ids[4 * new_tet_nr + TET_FACES[l][1]]]
                    tp2 = verts[tet_ids[4 * new_tet_nr + TET_FACES[l][2]]]
                    new_n = np.cross(tp1 - tp0, tp2 - tp0)
                    norm = np.linalg.norm(new_n)
                    if norm > 0:
                        new_n = new_n / norm
                    planes_n[4 * new_tet_nr + l] = new_n
                    planes_d[4 * new_tet_nr + l] = np.dot(new_n, tp0)

                if id0 < id1:
                    edges.append((id0, id1, new_tet_nr, 1))
                else:
                    edges.append((id1, id0, new_tet_nr, 1))

                if id1 < id2:
                    edges.append((id1, id2, new_tet_nr, 2))
                else:
                    edges.append((id2, id1, new_tet_nr, 2))

                if id2 < id0:
                    edges.append((id2, id0, new_tet_nr, 3))
                else:
                    edges.append((id0, id2, new_tet_nr, 3))

        # Fix neighbors
        sorted_edges = sorted(edges, key=cmp_to_key(_compare_edges))
        nr = 0
        num_edges = len(sorted_edges)

        while nr < num_edges:
            e0 = sorted_edges[nr]
            nr += 1

            if nr < num_edges and _equal_edges(sorted_edges[nr], e0):
                e1 = sorted_edges[nr]
                neighbors[4 * e0[2] + e0[3]] = e1[2]
                neighbors[4 * e1[2] + e1[3]] = e0[2]
                nr += 1

    # Remove outer, deleted and low-quality tets
    num_tets = len(tet_ids) // 4
    num_bad = 0
    result = []

    for i in range(num_tets):
        id0 = tet_ids[4 * i]
        id1 = tet_ids[4 * i + 1]
        id2 = tet_ids[4 * i + 2]
        id3 = tet_ids[4 * i + 3]

        # Skip deleted tets or tets containing big tet vertices
        if id0 < 0 or id0 >= first_big or id1 >= first_big or id2 >= first_big or id3 >= first_big:
            continue

        p0 = verts[id0]
        p1 = verts[id1]
        p2 = verts[id2]
        p3 = verts[id3]

        quality = tet_quality(p0, p1, p2, p3)
        if quality < min_quality:
            num_bad += 1
            continue

        result.extend([id0, id1, id2, id3])

    print(f"{num_bad} bad tets deleted")
    print(f"{len(result) // 4} tets created")

    return result


def tetrahedralize(points: np.ndarray, min_quality: float = 0.001) -> np.ndarray:
    """
    Compute the Delaunay tetrahedralization of a point cloud.

    This implements an incremental Delaunay tetrahedralization algorithm.
    Points are inserted one at a time, and the Delaunay condition is maintained
    by flipping tetrahedra that violate the empty circumsphere property.

    Args:
        points: Nx3 numpy array of 3D points
        min_quality: Minimum tetrahedron quality threshold (0-1, where 1 is
                    a perfect regular tetrahedron). Default is 0.001.

    Returns:
        Mx4 numpy array of tetrahedron vertex indices, where M is the number
        of tetrahedra and each row contains 4 indices into the input points array.

    Example:
        >>> points = np.random.rand(100, 3)
        >>> tets = tetrahedralize(points)
        >>> print(f"Created {len(tets)} tetrahedra from {len(points)} points")
    """
    if points.shape[0] < 4:
        raise ValueError("Need at least 4 points to create a tetrahedralization")

    if points.shape[1] != 3:
        raise ValueError("Points must be Nx3 array")

    # Add small random perturbation to avoid degeneracies
    verts = []
    for i in range(len(points)):
        p = points[i].copy()
        p[0] += _rand_eps()
        p[1] += _rand_eps()
        p[2] += _rand_eps()
        verts.append(p)

    # Compute bounding sphere
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))

    # Add big tetrahedron vertices that contain all points
    s = 5.0 * radius
    verts.append(np.array([-s, 0.0, -s]))
    verts.append(np.array([s, 0.0, -s]))
    verts.append(np.array([0.0, s, s]))
    verts.append(np.array([0.0, -s, s]))

    # Run tetrahedralization
    tet_ids = _create_tet_ids(verts, min_quality)

    # Convert to numpy array and reshape to Mx4
    if len(tet_ids) == 0:
        return np.zeros((0, 4), dtype=np.int32)

    return np.array(tet_ids, dtype=np.int32).reshape(-1, 4)
