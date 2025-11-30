#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/set.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <set>
#include <map>
#include <sstream>
#include <iomanip>

#include "delaunay.h"
#include "inputPLC.h"
#include "PLC.h"
#include <manifold/manifold.h>
#include <manifold/linalg.h>
#include "voro++.hh"

#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/halfedge_factories.h"



namespace nb = nanobind;
using namespace nb::literals;
using namespace manifold;

// ============================================================================
// Helper structures and functions
// ============================================================================

namespace {

const int tetFaces[4][3] = {{2,1,0}, {0,1,3}, {1,2,3}, {2,0,3}};

double randEps() {
    constexpr double eps = 0.000001;
    return -eps + 2.0 * ((double)rand() / RAND_MAX) * eps;
}

vec3 getCircumCenter(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3) {
    vec3 b = p1 - p0;
    vec3 c = p2 - p0;
    vec3 d = p3 - p0;

    double det = 2.0 * (b.x * (c.y * d.z - c.z * d.y) -
                        b.y * (c.x * d.z - c.z * d.x) +
                        b.z * (c.x * d.y - c.y * d.x));
    if (det == 0.0) return p0;

    vec3 v = linalg::cross(c, d) * linalg::dot(b, b) + linalg::cross(d, b) * linalg::dot(c, c) + linalg::cross(b, c) * linalg::dot(d, d);
    return p0 + v / det;
}

double tetQuality(const vec3& p0, const vec3& p1, const vec3& p2, const vec3& p3) {
    vec3 d0 = p1 - p0, d1 = p2 - p0, d2 = p3 - p0;
    vec3 d3 = p2 - p1, d4 = p3 - p2, d5 = p1 - p3;

    double s0 = linalg::length(d0), s1 = linalg::length(d1), s2 = linalg::length(d2);
    double s3 = linalg::length(d3), s4 = linalg::length(d4), s5 = linalg::length(d5);

    double ms = (s0*s0 + s1*s1 + s2*s2 + s3*s3 + s4*s4 + s5*s5) / 6.0;
    double rms = std::sqrt(ms);

    if (rms == 0.0) return 0.0;

    double s = 12.0 / std::sqrt(2.0);
    double vol = linalg::dot(d0, linalg::cross(d1,d2)) / 6.0;
    return s * vol / (rms * rms * rms);
}

struct Edge {
    int id0, id1, tetNr, faceNr;
    bool operator<(const Edge& o) const {
        return id0 < o.id0 || (id0 == o.id0 && id1 < o.id1);
    }
    bool operator==(const Edge& o) const { return id0 == o.id0 && id1 == o.id1; }
};

std::vector<uint32_t> createTetIds(std::vector<vec3>& verts, double minQuality) {
    std::vector<int> tetIds;
    std::vector<int> neighbors;
    std::vector<int> tetMarks;
    int tetMark = 0;
    int firstFreeTet = -1;

    std::vector<vec3> planesN;
    std::vector<double> planesD;

    int firstBig = (int)verts.size() - 4;

    // First big tet
    tetIds.push_back(firstBig);
    tetIds.push_back(firstBig + 1);
    tetIds.push_back(firstBig + 2);
    tetIds.push_back(firstBig + 3);
    tetMarks.push_back(0);

    for (int i = 0; i < 4; i++) {
        neighbors.push_back(-1);
        vec3 p0 = verts[firstBig + tetFaces[i][0]];
        vec3 p1 = verts[firstBig + tetFaces[i][1]];
        vec3 p2 = verts[firstBig + tetFaces[i][2]];
        vec3 n = linalg::cross(p1 - p0, p2 - p0);
				n = linalg::normalize(n);
        planesN.push_back(n);
        planesD.push_back(linalg::dot(p0, n));
    }

    // Insert each point
    for (int i = 0; i < firstBig; i++) {
        vec3 p = verts[i];

        // Find non-deleted tet
        int tetNr = 0;
        while (tetIds[4 * tetNr] < 0) tetNr++;

        // Find containing tet
        tetMark++;
        bool found = false;

        while (!found) {
            if (tetNr < 0 || tetMarks[tetNr] == tetMark) break;
            tetMarks[tetNr] = tetMark;

            int id0 = tetIds[4 * tetNr];
            int id1 = tetIds[4 * tetNr + 1];
            int id2 = tetIds[4 * tetNr + 2];
            int id3 = tetIds[4 * tetNr + 3];

            vec3 center = (verts[id0] + verts[id1] + verts[id2] + verts[id3]) * 0.25;

            double minT = std::numeric_limits<double>::infinity();
            int minFaceNr = -1;

            for (int j = 0; j < 4; j++) {
                vec3 n = planesN[4 * tetNr + j];
                double d = planesD[4 * tetNr + j];

                double hp = linalg::dot(n, p) - d;
                double hc = linalg::dot(n, center) - d;
                double t = hp - hc;
                if (t == 0) continue;

                t = -hc / t;
                if (t >= 0.0 && t < minT) {
                    minT = t;
                    minFaceNr = j;
                }
            }

            if (minT >= 1.0) found = true;
            else tetNr = neighbors[4 * tetNr + minFaceNr];
        }

        if (!found) continue;

        // Find violating tets
        tetMark++;
        std::vector<int> violatingTets;
        std::vector<int> stack = {tetNr};

        while (!stack.empty()) {
            tetNr = stack.back();
            stack.pop_back();
            if (tetMarks[tetNr] == tetMark) continue;
            tetMarks[tetNr] = tetMark;
            violatingTets.push_back(tetNr);

            for (int j = 0; j < 4; j++) {
                int n = neighbors[4 * tetNr + j];
                if (n < 0 || tetMarks[n] == tetMark) continue;

                int id0 = tetIds[4 * n];
                int id1 = tetIds[4 * n + 1];
                int id2 = tetIds[4 * n + 2];
                int id3 = tetIds[4 * n + 3];

                vec3 c = getCircumCenter(verts[id0], verts[id1], verts[id2], verts[id3]);
                double r = linalg::length(verts[id0] - c);
                if (linalg::length(p - c) < r) stack.push_back(n);
            }
        }

        // Remove old tets, create new ones
        std::vector<Edge> edges;

        for (int j = 0; j < (int)violatingTets.size(); j++) {
            tetNr = violatingTets[j];

            int ids[4], ns[4];
            for (int k = 0; k < 4; k++) {
                ids[k] = tetIds[4 * tetNr + k];
                ns[k] = neighbors[4 * tetNr + k];
            }

            tetIds[4 * tetNr] = -1;
            tetIds[4 * tetNr + 1] = firstFreeTet;
            firstFreeTet = tetNr;

            for (int k = 0; k < 4; k++) {
                int n = ns[k];
                if (n >= 0 && tetMarks[n] == tetMark) continue;

                int newTetNr = firstFreeTet;
                if (newTetNr >= 0) {
                    firstFreeTet = tetIds[4 * firstFreeTet + 1];
                } else {
                    newTetNr = (int)tetIds.size() / 4;
                    tetMarks.push_back(0);
                    for (int l = 0; l < 4; l++) {
                        tetIds.push_back(-1);
                        neighbors.push_back(-1);
                        planesN.push_back(vec3());
                        planesD.push_back(0.0);
                    }
                }

                int id0 = ids[tetFaces[k][2]];
                int id1 = ids[tetFaces[k][1]];
                int id2 = ids[tetFaces[k][0]];

                tetIds[4 * newTetNr] = id0;
                tetIds[4 * newTetNr + 1] = id1;
                tetIds[4 * newTetNr + 2] = id2;
                tetIds[4 * newTetNr + 3] = i;

                neighbors[4 * newTetNr] = n;
                if (n >= 0) {
                    for (int l = 0; l < 4; l++) {
                        if (neighbors[4 * n + l] == tetNr)
                            neighbors[4 * n + l] = newTetNr;
                    }
                }

                neighbors[4 * newTetNr + 1] = -1;
                neighbors[4 * newTetNr + 2] = -1;
                neighbors[4 * newTetNr + 3] = -1;

                for (int l = 0; l < 4; l++) {
                    vec3 fp0 = verts[tetIds[4 * newTetNr + tetFaces[l][0]]];
                    vec3 fp1 = verts[tetIds[4 * newTetNr + tetFaces[l][1]]];
                    vec3 fp2 = verts[tetIds[4 * newTetNr + tetFaces[l][2]]];
                    vec3 newN = linalg::cross(fp1 - fp0, fp2 - fp0);
                    newN = linalg::normalize(newN);
                    planesN[4 * newTetNr + l] = newN;
                    planesD[4 * newTetNr + l] = linalg::dot(newN, fp0);
                }

                Edge e;
                e.tetNr = newTetNr;
                if (id0 < id1) { e.id0 = id0; e.id1 = id1; e.faceNr = 1; }
                else { e.id0 = id1; e.id1 = id0; e.faceNr = 1; }
                edges.push_back(e);

                if (id1 < id2) { e.id0 = id1; e.id1 = id2; e.faceNr = 2; }
                else { e.id0 = id2; e.id1 = id1; e.faceNr = 2; }
                edges.push_back(e);

                if (id2 < id0) { e.id0 = id2; e.id1 = id0; e.faceNr = 3; }
                else { e.id0 = id0; e.id1 = id2; e.faceNr = 3; }
                edges.push_back(e);
            }
        }

        // Fix neighbors
        std::sort(edges.begin(), edges.end());

        size_t nr = 0;
        while (nr < edges.size()) {
            Edge e0 = edges[nr++];
            if (nr < edges.size() && edges[nr] == e0) {
                Edge e1 = edges[nr++];
                neighbors[4 * e0.tetNr + e0.faceNr] = e1.tetNr;
                neighbors[4 * e1.tetNr + e1.faceNr] = e0.tetNr;
            }
        }
    }

    // Remove outer, deleted, and outside tets
    int numTets = (int)tetIds.size() / 4;
    std::vector<uint32_t> result;

    for (int i = 0; i < numTets; i++) {
        int id0 = tetIds[4 * i];
        int id1 = tetIds[4 * i + 1];
        int id2 = tetIds[4 * i + 2];
        int id3 = tetIds[4 * i + 3];

        if (id0 < 0 || id0 >= firstBig || id1 >= firstBig || id2 >= firstBig || id3 >= firstBig)
            continue;

        vec3 p0 = verts[id0], p1 = verts[id1], p2 = verts[id2], p3 = verts[id3];

        double quality = tetQuality(p0, p1, p2, p3);
        if (quality < minQuality) continue;

        result.push_back((uint32_t)id0);
        result.push_back((uint32_t)id1);
        result.push_back((uint32_t)id2);
        result.push_back((uint32_t)id3);
    }

    return result;
}

} // anonymous namespace

// ============================================================================
// Core C++ functions (extracted from lambda bindings)
// ============================================================================

// Constrained Delaunay tetrahedrization
std::tuple<std::vector<std::array<double, 3>>, std::vector<std::array<uint32_t, 4>>>
tetrahedrize(double* points_data, size_t num_points,
             uint32_t* triangles_data, size_t num_triangles) {

    inputPLC plc;
    plc.initFromVectors(points_data, num_points, triangles_data, num_triangles, true);

    TetMesh* tin = new TetMesh;
    tin->init_vertices(plc.coordinates.data(), plc.numVertices());
    tin->tetrahedrize();
    tin->optimizeNearDegenerateTets(false);

    PLCx Steiner_plc(*tin, plc.triangle_vertices.data(), plc.numTriangles());
    Steiner_plc.segmentRecovery_HSi(false);
    Steiner_plc.faceRecovery(false);
    Steiner_plc.markInnerTets();

    std::vector<std::array<double, 3>> output_points;
    std::vector<std::array<uint32_t, 4>> output_tetrahedra;

    for (const auto& vertex : tin->vertices) {
        double coords[3];
        if (vertex->getApproxXYZCoordinates(coords[0], coords[1], coords[2])) {
            output_points.push_back({coords[0], coords[1], coords[2]});
        } else {
            delete tin;
            throw std::runtime_error("Vertex has non-finite coordinates");
        }
    }

    for (uint32_t i = 0; i < tin->numTets(); i++) {
        if (tin->mark_tetrahedra[i] == DT_IN) {
            output_tetrahedra.push_back({
                tin->tet_node[(i * 4)],
                tin->tet_node[(i * 4) + 1],
                tin->tet_node[(i * 4) + 2],
                tin->tet_node[(i * 4) + 3]
            });
        }
    }
    delete tin;
    return std::make_tuple(output_points, output_tetrahedra);
}

// Unconstrained Delaunay tetrahedrization
std::vector<std::array<uint32_t, 4>>
createTets(const double* points_data, size_t num_points, double minQuality) {

    if (num_points < 4) {
        throw std::invalid_argument("Need at least 4 points to create a tetrahedralization");
    }

    std::vector<vec3> tetVerts;
    for (size_t i = 0; i < num_points; i++) {
        tetVerts.push_back(vec3(
            points_data[i * 3 + 0] + randEps(),
            points_data[i * 3 + 1] + randEps(),
            points_data[i * 3 + 2] + randEps()
        ));
    }

    vec3 center(0, 0, 0);
    for (const auto& p : tetVerts) {
        center += p;
    }
    center /= (double)tetVerts.size();

    double radius = 0.0;
    for (const auto& p : tetVerts) {
        double d = linalg::length(p - center);
        radius = std::max(radius, d);
    }

    double s = 5.0 * radius;
    tetVerts.push_back(vec3(-s, 0.0, -s));
    tetVerts.push_back(vec3(s, 0.0, -s));
    tetVerts.push_back(vec3(0.0, s, s));
    tetVerts.push_back(vec3(0.0, -s, s));

    std::vector<uint32_t> faces = createTetIds(tetVerts, minQuality);
    int numTets = (int)faces.size() / 4;

    std::vector<std::array<uint32_t, 4>> output_tetrahedra;
    for (int i = 0; i < numTets; i++) {
        output_tetrahedra.push_back({
            faces[4 * i],
            faces[4 * i + 1],
            faces[4 * i + 2],
            faces[4 * i + 3]
        });
    }

    return output_tetrahedra;
}

// Create cube manifold
std::tuple<std::vector<double>, std::vector<size_t>>
createCube(double x, double y, double z) {
    manifold::Manifold cube = manifold::Manifold::Cube({x, y, z});
    manifold::MeshGL64 mesh = cube.GetMeshGL64();
    return std::make_tuple(mesh.vertProperties, mesh.triVerts);
}

// Create sphere manifold
std::tuple<std::vector<double>, std::vector<size_t>>
createSphere(double radius) {
    manifold::Manifold sphere = manifold::Manifold::Sphere(radius);
    manifold::MeshGL64 mesh = sphere.GetMeshGL64();
    return std::make_tuple(mesh.vertProperties, mesh.triVerts);
}

// Compute Voronoi diagram
std::vector<std::vector<double>>
voronoi3d(const double* points_data, size_t num_points,
          const double* wts_data, size_t num_wts,
          const double* bounds_data) {

    double min_x = bounds_data[0] - 0.1, max_x = bounds_data[1] + 0.1;
    double min_y = bounds_data[2] - 0.1, max_y = bounds_data[3] + 0.1;
    double min_z = bounds_data[4] - 0.1, max_z = bounds_data[5] + 0.1;

    if (num_points == 0) {
        throw std::invalid_argument("Points array cannot be empty");
    }

    double V = (max_x - min_x) * (max_y - min_y) * (max_z - min_z);
    double Nthird = std::pow((double)num_points / V, 1.0 / 3.0);

    voro::container_poly container(
        bounds_data[0], bounds_data[1], bounds_data[2],
        bounds_data[3], bounds_data[4], bounds_data[5],
        std::max(1, (int)std::round(Nthird * (max_x - min_x))),
        std::max(1, (int)std::round(Nthird * (max_y - min_y))),
        std::max(1, (int)std::round(Nthird * (max_z - min_z))),
        false, false, false, (int)num_points);

    bool hasWeights = num_wts == num_points;
    for (size_t i = 0; i < num_points; i++) {
        container.put(i, points_data[i * 3], points_data[i * 3 + 1], points_data[i * 3 + 2],
                      hasWeights ? wts_data[i] : 1.0);
    }

    std::vector<std::vector<double>> cells;
    voro::c_loop_all vl(container);
    if (vl.start()) do {
        int id;
        double x, y, z, r;
        vl.pos(id, x, y, z, r);

        voro::voronoicell_neighbor c;
        if (container.compute_cell(c, vl)) {
            std::vector<double> verts;
            verts.reserve(c.p * 3);
            for (int i = 0; i < c.p; i++) {
                verts.push_back(x + 0.5 * c.pts[(4 * i) + 0]);
                verts.push_back(y + 0.5 * c.pts[(4 * i) + 1]);
                verts.push_back(z + 0.5 * c.pts[(4 * i) + 2]);
            }
            cells.push_back(verts);
        }
    } while (vl.inc());

    return cells;
}

// Get reflex faces from a mesh
std::set<uint32_t> getReflexFaces(const double* verts_data, size_t num_verts,
                                   const uint32_t* tris_data, size_t num_tris) {
    std::vector<std::vector<size_t>> polygons;
    std::vector<geometrycentral::Vector3> vertexPositions;

    for (size_t i = 0; i < num_verts; i++) {
        vertexPositions.push_back(geometrycentral::Vector3(
            verts_data[i * 3], verts_data[i * 3 + 1], verts_data[i * 3 + 2]));
    }

    for (size_t i = 0; i < num_tris; i++) {
        polygons.push_back({tris_data[i * 3], tris_data[i * 3 + 1], tris_data[i * 3 + 2]});
    }

    auto meshAndGeo = geometrycentral::surface::makeHalfedgeAndGeometry(polygons, vertexPositions);
    auto mesh = std::move(std::get<0>(meshAndGeo));
    auto geometry = std::move(std::get<1>(meshAndGeo));

    std::set<uint32_t> reflexFaces;
    for (auto e : mesh->edges()) {
        auto he1 = e.halfedge();
        auto he2 = he1.twin();
        auto f1 = he1.face();
        auto f2 = he2.face();
        auto n1 = geometry->faceNormal(f1);
        auto n2 = geometry->faceNormal(f2);
        geometrycentral::Vector3 tangent = geometrycentral::cross(n1,
            geometry->vertexPositions[e.secondVertex().getIndex()] -
            geometry->vertexPositions[e.firstVertex().getIndex()]);
        double tangentProjection = geometrycentral::dot(n2, tangent);
        if (tangentProjection > 0.000000001) {
            reflexFaces.insert((uint32_t)f1.getIndex());
            reflexFaces.insert((uint32_t)f2.getIndex());
        }
    }
    return reflexFaces;
}

// Get reflex edges (original function)
std::vector<std::tuple<std::array<size_t, 2>, std::array<size_t, 2>>>
getReflexEdges(const double* points_data, size_t num_points,
               const uint32_t* triangles_data, size_t num_triangles) {

    std::vector<std::vector<size_t>> polygons;
    std::vector<geometrycentral::Vector3> vertexPositions;

    for (size_t i = 0; i < num_points; i++) {
        vertexPositions.push_back(geometrycentral::Vector3(
            points_data[i * 3], points_data[i * 3 + 1], points_data[i * 3 + 2]));
    }

    for (size_t i = 0; i < num_triangles; i++) {
        polygons.push_back({triangles_data[i * 3], triangles_data[i * 3 + 1], triangles_data[i * 3 + 2]});
    }

    auto meshAndGeo = geometrycentral::surface::makeHalfedgeAndGeometry(polygons, vertexPositions);
    auto mesh = std::move(std::get<0>(meshAndGeo));
    auto geometry = std::move(std::get<1>(meshAndGeo));

    std::vector<std::tuple<std::array<size_t, 2>, std::array<size_t, 2>>> reflexEdges;
    for (auto e : mesh->edges()) {
        auto he1 = e.halfedge();
        auto he2 = he1.twin();
        auto f1 = he1.face();
        auto f2 = he2.face();
        auto n1 = geometry->faceNormal(f1);
        auto n2 = geometry->faceNormal(f2);
        geometrycentral::Vector3 tangent = geometrycentral::cross(n1,
            geometry->vertexPositions[e.secondVertex().getIndex()] -
            geometry->vertexPositions[e.firstVertex().getIndex()]);
        double tangentProjection = geometrycentral::dot(n2, tangent);
        if (tangentProjection > 0.000000001) {
            size_t v1 = e.halfedge().vertex().getIndex();
            size_t v2 = e.halfedge().twin().vertex().getIndex();
            reflexEdges.push_back(std::make_tuple(
                std::array<size_t, 2>{v1, v2},
                std::array<size_t, 2>{f1.getIndex(), f2.getIndex()}));
        }
    }
    return reflexEdges;
}

// Helper: Create a durable face key for comparison
std::string getDurableFaceKey(const vec3& v0, const vec3& v1, const vec3& v2) {
    std::vector<vec3> sorted = {v0, v1, v2};
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream oss;
    oss << std::setprecision(10);
    for (const auto& v : sorted) {
        oss << v.x << "," << v.y << "," << v.z << ";";
    }
    return oss.str();
}

// Convert Voronoi cells to manifold hulls
std::vector<manifold::Manifold> cellsToManifolds(const std::vector<std::vector<double>>& cells) {
    std::vector<manifold::Manifold> hulls;
    for (const auto& cell : cells) {
        std::vector<vec3> points;
        for (size_t i = 0; i < cell.size(); i += 3) {
            points.push_back(vec3(cell[i], cell[i + 1], cell[i + 2]));
        }
        if (points.size() >= 4) {
            hulls.push_back(manifold::Manifold::Hull(points));
        }
    }
    return hulls;
}

// ============================================================================
// mmf_tetrahedron_convex_decomposition
// ============================================================================

std::vector<manifold::Manifold> mmfTetrahedronConvexDecomposition(manifold::Manifold shape) {
    std::vector<manifold::Manifold> outputs;

    if (shape.IsEmpty()) {
        return outputs;
    }

    std::vector<manifold::Manifold> shapes = shape.Decompose();
    if (shapes.empty()) {
        outputs.push_back(shape);
        return outputs;
    }

    for (auto& curShape : shapes) {
        if (curShape.IsEmpty()) continue;

        manifold::MeshGL64 meshGL = curShape.GetMeshGL64();
        const auto& vertProps = meshGL.vertProperties;
        const auto& triVerts = meshGL.triVerts;

        size_t numVerts = vertProps.size() / meshGL.numProp;
        size_t numTris = triVerts.size() / 3;

        // Extract vertices
        std::vector<double> verts;
        for (size_t i = 0; i < numVerts; i++) {
            verts.push_back(vertProps[i * meshGL.numProp + 0]);
            verts.push_back(vertProps[i * meshGL.numProp + 1]);
            verts.push_back(vertProps[i * meshGL.numProp + 2]);
        }

        // Extract triangles
        std::vector<uint32_t> tris;
        for (size_t i = 0; i < triVerts.size(); i++) {
            tris.push_back((uint32_t)triVerts[i]);
        }

        // Get reflex faces
        std::set<uint32_t> reflexFaceSet = getReflexFaces(verts.data(), numVerts, tris.data(), numTris);

        if (reflexFaceSet.empty()) {
            outputs.push_back(curShape);
            continue;
        }

        // Build durable reflex face keys and collect face vertices
        std::set<std::string> durableReflexFaceKeys;
        std::vector<vec3> uniqueFaceVertices;
        std::set<std::string> addedVerts;

        for (uint32_t faceIdx : reflexFaceSet) {
            uint32_t i0 = tris[faceIdx * 3 + 0];
            uint32_t i1 = tris[faceIdx * 3 + 1];
            uint32_t i2 = tris[faceIdx * 3 + 2];

            vec3 v0(verts[i0 * 3], verts[i0 * 3 + 1], verts[i0 * 3 + 2]);
            vec3 v1(verts[i1 * 3], verts[i1 * 3 + 1], verts[i1 * 3 + 2]);
            vec3 v2(verts[i2 * 3], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);

            durableReflexFaceKeys.insert(getDurableFaceKey(v0, v1, v2));

            // Add unique vertices
            std::string k0 = std::to_string(v0.x) + "," + std::to_string(v0.y) + "," + std::to_string(v0.z);
            std::string k1 = std::to_string(v1.x) + "," + std::to_string(v1.y) + "," + std::to_string(v1.z);
            std::string k2 = std::to_string(v2.x) + "," + std::to_string(v2.y) + "," + std::to_string(v2.z);
            if (addedVerts.find(k0) == addedVerts.end()) { addedVerts.insert(k0); uniqueFaceVertices.push_back(v0); }
            if (addedVerts.find(k1) == addedVerts.end()) { addedVerts.insert(k1); uniqueFaceVertices.push_back(v1); }
            if (addedVerts.find(k2) == addedVerts.end()) { addedVerts.insert(k2); uniqueFaceVertices.push_back(v2); }
        }

        if (uniqueFaceVertices.empty()) {
            outputs.push_back(curShape);
            continue;
        }

        // Compute bounds
        vec3 minBounds(std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity());
        vec3 maxBounds(-std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity());

        for (const auto& v : uniqueFaceVertices) {
            minBounds.x = std::min(minBounds.x, v.x);
            minBounds.y = std::min(minBounds.y, v.y);
            minBounds.z = std::min(minBounds.z, v.z);
            maxBounds.x = std::max(maxBounds.x, v.x);
            maxBounds.y = std::max(maxBounds.y, v.y);
            maxBounds.z = std::max(maxBounds.z, v.z);
        }

        minBounds.x -= 0.1; minBounds.y -= 0.1; minBounds.z -= 0.1;
        maxBounds.x += 0.1; maxBounds.y += 0.1; maxBounds.z += 0.1;

        // Add cube corners
        std::vector<vec3> tetVertices = uniqueFaceVertices;
        tetVertices.push_back(vec3(minBounds.x, minBounds.y, minBounds.z));
        tetVertices.push_back(vec3(maxBounds.x, minBounds.y, minBounds.z));
        tetVertices.push_back(vec3(minBounds.x, maxBounds.y, minBounds.z));
        tetVertices.push_back(vec3(maxBounds.x, maxBounds.y, minBounds.z));
        tetVertices.push_back(vec3(minBounds.x, minBounds.y, maxBounds.z));
        tetVertices.push_back(vec3(maxBounds.x, minBounds.y, maxBounds.z));
        tetVertices.push_back(vec3(minBounds.x, maxBounds.y, maxBounds.z));
        tetVertices.push_back(vec3(maxBounds.x, maxBounds.y, maxBounds.z));

        // Convert to flat array for createTets
        std::vector<double> tetVertsFlat;
        for (const auto& v : tetVertices) {
            tetVertsFlat.push_back(v.x);
            tetVertsFlat.push_back(v.y);
            tetVertsFlat.push_back(v.z);
        }

        // Tetrahedralize
        auto tetIndices = createTets(tetVertsFlat.data(), tetVertices.size(), 0.001);

        // Compute circumcenters for tets that share a face with reflex faces
        std::map<std::string, std::pair<vec3, double>> circumcenters;

        for (const auto& tetIdx : tetIndices) {
            vec3 tv0 = tetVertices[tetIdx[0]];
            vec3 tv1 = tetVertices[tetIdx[1]];
            vec3 tv2 = tetVertices[tetIdx[2]];
            vec3 tv3 = tetVertices[tetIdx[3]];

            // Check if tet shares a face with a reflex face
            bool faceFound = false;
            vec3 tetFaceVerts[4][3] = {
                {tv1, tv2, tv3}, {tv0, tv2, tv3}, {tv0, tv1, tv3}, {tv0, tv1, tv2}
            };

            for (int f = 0; f < 4; f++) {
                std::string key = getDurableFaceKey(tetFaceVerts[f][0], tetFaceVerts[f][1], tetFaceVerts[f][2]);
                if (durableReflexFaceKeys.count(key) > 0) {
                    faceFound = true;
                    break;
                }
            }

            if (!faceFound) continue;

            vec3 cc = getCircumCenter(tv0, tv1, tv2, tv3);
            double r = linalg::length(cc - tv0);
            std::ostringstream oss;
            oss << std::setprecision(10) << cc.x << "," << cc.y << "," << cc.z;
            circumcenters[oss.str()] = std::make_pair(cc, r);
        }

        if (circumcenters.empty()) {
            outputs.push_back(curShape);
            continue;
        }

        // Build points and weights arrays for Voronoi
        std::vector<double> points;
        std::vector<double> wts;
        for (const auto& kv : circumcenters) {
            points.push_back(kv.second.first.x);
            points.push_back(kv.second.first.y);
            points.push_back(kv.second.first.z);
            wts.push_back(kv.second.second);
        }

        double bounds[6] = {minBounds.x, maxBounds.x, minBounds.y, maxBounds.y, minBounds.z, maxBounds.z};

        // Compute Voronoi
        auto cells = voronoi3d(points.data(), points.size() / 3, wts.data(), wts.size(), bounds);
        auto hulls = cellsToManifolds(cells);

        // Intersect hulls with original shape
        std::vector<manifold::Manifold> intersectedHulls;
        for (auto& hull : hulls) {
            manifold::Manifold intersected = shape ^ hull;
            if (!intersected.IsEmpty()) {
                intersectedHulls.push_back(intersected);
            }
        }

        // Decompose and simplify
        for (int iter = 0; iter < 3; iter++) {
            std::vector<manifold::Manifold> decomposed;
            for (auto& hull : intersectedHulls) {
                auto parts = hull.Decompose();
                for (auto& part : parts) {
                    decomposed.push_back(part.AsOriginal().Simplify(0.0000001));
                }
            }
            intersectedHulls = decomposed;
        }

        // Final decomposition
        std::vector<manifold::Manifold> finalHulls;
        for (auto& hull : intersectedHulls) {
            auto parts = hull.Decompose();
            for (auto& part : parts) {
                if (!part.IsEmpty()) {
                    finalHulls.push_back(part);
                }
            }
        }

        // Recurse on non-convex pieces
        for (size_t i = 0; i < finalHulls.size(); i++) {
            auto& hull = finalHulls[i];
            manifold::MeshGL64 hullMesh = hull.GetMeshGL64();
            size_t hNumVerts = hullMesh.vertProperties.size() / hullMesh.numProp;
            size_t hNumTris = hullMesh.triVerts.size() / 3;

            std::vector<double> hVerts;
            for (size_t j = 0; j < hNumVerts; j++) {
                hVerts.push_back(hullMesh.vertProperties[j * hullMesh.numProp + 0]);
                hVerts.push_back(hullMesh.vertProperties[j * hullMesh.numProp + 1]);
                hVerts.push_back(hullMesh.vertProperties[j * hullMesh.numProp + 2]);
            }
            std::vector<uint32_t> hTris;
            for (size_t j = 0; j < hullMesh.triVerts.size(); j++) {
                hTris.push_back((uint32_t)hullMesh.triVerts[j]);
            }

            auto hReflexFaces = getReflexFaces(hVerts.data(), hNumVerts, hTris.data(), hNumTris);
            manifold::Manifold enforcedHull = hull.Hull();

            if (!hReflexFaces.empty() && (enforcedHull.Volume() - hull.Volume() > 0.0001)) {
                // Rotate and recurse
                manifold::Manifold rotated = hull.Rotate(90, 0, 0);
                auto recursed = mmfTetrahedronConvexDecomposition(rotated);
                for (auto& r : recursed) {
                    outputs.push_back(r.Rotate(-90, 0, 0));
                }
            } else {
                outputs.push_back(hull);
            }
        }
    }

    return outputs;
}

// ============================================================================
// Python bindings
// ============================================================================

NB_MODULE(vocd_ext, m) {
    m.doc() = "Geometry tools using CDT, Manifold, and Voro++";

    // CDT: 3D Constrained Delaunay Tetrahedrization
    m.def("tetrahedrize", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        nb::ndarray<uint32_t, nb::shape<-1, 3>> triangles) {
        return tetrahedrize(points.data(), points.shape(0),
                           triangles.data(), triangles.shape(0));
    }, "points"_a, "triangles"_a,
       "Perform 3D constrained Delaunay tetrahedrization on a set of points and triangles");

    // Unconstrained Delaunay tetrahedrization
    m.def("createTets", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        double minQuality) {
        return createTets(points.data(), points.shape(0), minQuality);
    }, "points"_a, "minQuality"_a = 0.001,
       "Compute the Delaunay tetrahedralization of a point cloud. "
       "minQuality filters out degenerate tetrahedra (1.0 = perfect, 0.001 = default).");

    // Manifold: Create a simple cube
    m.def("create_cube", [](double x, double y, double z) {
        return createCube(x, y, z);
    }, "x"_a, "y"_a, "z"_a, "Create a cube with given dimensions");

    // Manifold: Create a simple sphere
    m.def("create_sphere", [](double radius) {
        return createSphere(radius);
    }, "radius"_a, "Create a sphere with given radius");

    // Voro++: Compute Voronoi diagram
    m.def("voronoi_3d", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        nb::ndarray<double, nb::shape<-1>> wts,
        nb::ndarray<double, nb::shape<6>> bounds) {
        return voronoi3d(points.data(), points.shape(0),
                        wts.data(), wts.size(), bounds.data());
    }, "points"_a, "wts"_a, "bounds"_a,
       "Compute Voronoi cell volumes for 3D points within given bounds [x_min, x_max, y_min, y_max, z_min, z_max]");

    // Get reflex edges
    m.def("getReflexEdges", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        nb::ndarray<uint32_t, nb::shape<-1, 3>> triangles) {
        return getReflexEdges(points.data(), points.shape(0),
                             triangles.data(), triangles.shape(0));
    }, "points"_a, "triangles"_a,
       "Identify reflex edges in a triangular mesh defined by input triangles");

    // Get reflex faces
    m.def("getReflexFaces", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        nb::ndarray<uint32_t, nb::shape<-1, 3>> triangles) {
        return getReflexFaces(points.data(), points.shape(0),
                             triangles.data(), triangles.shape(0));
    }, "points"_a, "triangles"_a,
       "Identify reflex faces in a triangular mesh");

    // MMF Tetrahedron Convex Decomposition
    m.def("mmf_tetrahedron_convex_decomposition", [](
        nb::ndarray<double, nb::shape<-1, -1>> vertProperties,
        nb::ndarray<size_t, nb::shape<-1>> triVerts) {

        // Build manifold from mesh data
        manifold::MeshGL64 meshGL;
        meshGL.numProp = vertProperties.shape(1);
        meshGL.vertProperties.reserve(vertProperties.shape(0) * meshGL.numProp);
        for (size_t i = 0; i < vertProperties.shape(0); i++) {
            for (size_t j = 0; j < meshGL.numProp; j++) {
                meshGL.vertProperties.push_back(vertProperties(i, j));
            }
        }
        meshGL.triVerts.reserve(triVerts.shape(0));
        for (size_t i = 0; i < triVerts.shape(0); i++) {
            meshGL.triVerts.push_back(triVerts(i));
        }

        manifold::Manifold shape(meshGL);
        auto results = mmfTetrahedronConvexDecomposition(shape);

        // Convert results back to mesh data
        std::vector<std::tuple<std::vector<double>, std::vector<size_t>>> output;
        for (auto& result : results) {
            auto mesh = result.GetMeshGL64();
            output.push_back(std::make_tuple(mesh.vertProperties, mesh.triVerts));
        }
        return output;
    }, "vertProperties"_a, "triVerts"_a,
       "Perform convex decomposition using tetrahedron circumcenters and Voronoi cells");

    // Version info
    m.attr("__version__") = "0.1.0";
}
