#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>

#include "delaunay.h"
#include "inputPLC.h"
#include "PLC.h"
#include <manifold/manifold.h>
#include "voro++.hh"

#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/halfedge_factories.h"

namespace nb = nanobind;
using namespace nb::literals;

// Helper structures and functions for createTets
namespace {

const int tetFaces[4][3] = {{2,1,0}, {0,1,3}, {1,2,3}, {2,0,3}};

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(double s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(double s) const { return Vec3(x / s, y / s, z / s); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator/=(double s) { x /= s; y /= s; z /= s; return *this; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }
    double magnitude() const { return std::sqrt(x * x + y * y + z * z); }
    void normalize() {
        double m = magnitude();
        if (m > 0) { x /= m; y /= m; z /= m; }
    }
};

double randEps() {
    constexpr double eps = 0.000001;
    return -eps + 2.0 * ((double)rand() / RAND_MAX) * eps;
}

Vec3 getCircumCenter(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    Vec3 b = p1 - p0;
    Vec3 c = p2 - p0;
    Vec3 d = p3 - p0;

    double det = 2.0 * (b.x * (c.y * d.z - c.z * d.y) -
                        b.y * (c.x * d.z - c.z * d.x) +
                        b.z * (c.x * d.y - c.y * d.x));
    if (det == 0.0) return p0;

    Vec3 v = c.cross(d) * b.dot(b) + d.cross(b) * c.dot(c) + b.cross(c) * d.dot(d);
    return p0 + v / det;
}

double tetQuality(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3) {
    Vec3 d0 = p1 - p0, d1 = p2 - p0, d2 = p3 - p0;
    Vec3 d3 = p2 - p1, d4 = p3 - p2, d5 = p1 - p3;

    double s0 = d0.magnitude(), s1 = d1.magnitude(), s2 = d2.magnitude();
    double s3 = d3.magnitude(), s4 = d4.magnitude(), s5 = d5.magnitude();

    double ms = (s0*s0 + s1*s1 + s2*s2 + s3*s3 + s4*s4 + s5*s5) / 6.0;
    double rms = std::sqrt(ms);

    if (rms == 0.0) return 0.0;

    double s = 12.0 / std::sqrt(2.0);
    double vol = d0.dot(d1.cross(d2)) / 6.0;
    return s * vol / (rms * rms * rms);
}

struct Edge {
    int id0, id1, tetNr, faceNr;
    bool operator<(const Edge& o) const {
        return id0 < o.id0 || (id0 == o.id0 && id1 < o.id1);
    }
    bool operator==(const Edge& o) const { return id0 == o.id0 && id1 == o.id1; }
};

std::vector<uint32_t> createTetIds(std::vector<Vec3>& verts,
                                    //const std::vector<Vec3>& meshVerts,
                                    //const std::vector<std::array<uint32_t, 3>>& tris,
                                    double minQuality) {
    std::vector<int> tetIds;
    std::vector<int> neighbors;
    std::vector<int> tetMarks;
    int tetMark = 0;
    int firstFreeTet = -1;

    std::vector<Vec3> planesN;
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
        Vec3 p0 = verts[firstBig + tetFaces[i][0]];
        Vec3 p1 = verts[firstBig + tetFaces[i][1]];
        Vec3 p2 = verts[firstBig + tetFaces[i][2]];
        Vec3 n = (p1 - p0).cross(p2 - p0);
        n.normalize();
        planesN.push_back(n);
        planesD.push_back(p0.dot(n));
    }

    // Insert each point
    for (int i = 0; i < firstBig; i++) {
        Vec3 p = verts[i];

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

            Vec3 center = (verts[id0] + verts[id1] + verts[id2] + verts[id3]) * 0.25;

            double minT = std::numeric_limits<double>::infinity();
            int minFaceNr = -1;

            for (int j = 0; j < 4; j++) {
                Vec3 n = planesN[4 * tetNr + j];
                double d = planesD[4 * tetNr + j];

                double hp = n.dot(p) - d;
                double hc = n.dot(center) - d;
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

                Vec3 c = getCircumCenter(verts[id0], verts[id1], verts[id2], verts[id3]);
                double r = (verts[id0] - c).magnitude();
                if ((p - c).magnitude() < r) stack.push_back(n);
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
                        planesN.push_back(Vec3());
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
                    Vec3 fp0 = verts[tetIds[4 * newTetNr + tetFaces[l][0]]];
                    Vec3 fp1 = verts[tetIds[4 * newTetNr + tetFaces[l][1]]];
                    Vec3 fp2 = verts[tetIds[4 * newTetNr + tetFaces[l][2]]];
                    Vec3 newN = (fp1 - fp0).cross(fp2 - fp0);
                    newN.normalize();
                    planesN[4 * newTetNr + l] = newN;
                    planesD[4 * newTetNr + l] = newN.dot(fp0);
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

        Vec3 p0 = verts[id0], p1 = verts[id1], p2 = verts[id2], p3 = verts[id3];

        double quality = tetQuality(p0, p1, p2, p3);
        if (quality < minQuality) continue;

        Vec3 center = (p0 + p1 + p2 + p3) * 0.25;
        //if (!isInside(meshVerts, tris, center)) continue;

        result.push_back((uint32_t)id0);
        result.push_back((uint32_t)id1);
        result.push_back((uint32_t)id2);
        result.push_back((uint32_t)id3);
    }

    return result;
}

} // anonymous namespace

NB_MODULE(vocd_ext, m) {
    m.doc() = "Geometry tools using CDT, Manifold, and Voro++";

    // CDT Example: 3D Constrained Delaunay Tetrahedrization
    m.def("tetrahedrize", [](
        nb::ndarray<  double, nb::shape<-1, 3>> points,
        nb::ndarray<uint32_t, nb::shape<-1, 3>> triangles) {

        // Create a PLC from the input points and triangles
        inputPLC plc;
        plc.initFromVectors(
            points.data(), points.shape(0),
            triangles.data(), triangles.shape(0), true);

        //TetMesh* tin = createSteinerCDT(plc, std::string("v").c_str());

        //if (bbox) plc.addBoundingBoxVertices();

        // Build a delaunay tetrahedrization of the vertices
        TetMesh  *tin = new TetMesh;
        tin->init_vertices(plc.coordinates.data(), plc.numVertices());
        tin->tetrahedrize();

        tin->optimizeNearDegenerateTets(false);

        // Build a structured PLC linked to the Delaunay tetrahedrization
        PLCx Steiner_plc(*tin, plc.triangle_vertices.data(), plc.numTriangles());

        // Recover segments by inserting Steiner points in both the PLC and the tetrahedrization
        Steiner_plc.segmentRecovery_HSi(false);

        // Recover PLC faces by locally remeshing the tetrahedrization
        bool sisMethodWorks = Steiner_plc.faceRecovery(false);

        // Mark the tets which are bounded by the PLC.
        // If the PLC is not a valid polyhedron (i.e. it has odd-valency edges)
        // all the tets but the ghosts are marked as "internal".
        uint32_t num_inner_tets = (uint32_t)Steiner_plc.markInnerTets();

        std::vector<std::array<  double, 3>> output_points;
        std::vector<std::array<uint32_t, 4>> output_tetrahedra;

        // Output a list of points and tetrahedra indices
        for (const auto& vertex : tin->vertices) {
            double coords[3];
            if(vertex->getApproxXYZCoordinates(coords[0], coords[1], coords[2])){
                output_points.push_back({coords[0], coords[1], coords[2]});
            } else {
                throw std::runtime_error("Vertex has non-finite coordinates");
            }
        }

        for (uint32_t i = 0; i < tin->numTets(); i ++) {
            if (tin->mark_tetrahedra[i] == DT_IN) {
                output_tetrahedra.push_back({
                    tin->tet_node[(i * 4)    ],
                    tin->tet_node[(i * 4) + 1],
                    tin->tet_node[(i * 4) + 2],
                    tin->tet_node[(i * 4) + 3]
                });
            }
        }
        delete tin; // Clean up the TetMesh object
        return std::make_tuple(output_points, output_tetrahedra);
    }, "points"_a, "triangles"_a,
         "Perform 3D constrained Delaunay tetrahedrization on a set of points and triangles");

    // Unconstrained Delaunay tetrahedrization (ported from tetrahedralize.py)
    m.def("createTets", [](
        nb::ndarray<double, nb::shape<-1, 3>> points,
        double minQuality) {

        if (points.shape(0) < 4) {
            throw std::invalid_argument("Need at least 4 points to create a tetrahedralization");
        }

        // Build tetVerts: start with input points (with small random perturbation)
        std::vector<Vec3> tetVerts;
        for (size_t i = 0; i < points.shape(0); i++) {
            tetVerts.push_back(Vec3(
                points(i, 0) + randEps(),
                points(i, 1) + randEps(),
                points(i, 2) + randEps()
            ));
        }

        // Compute bounding sphere (center and radius)
        Vec3 center(0, 0, 0);
        for (const auto& p : tetVerts) {
            center += p;
        }
        center /= (double)tetVerts.size();

        double radius = 0.0;
        for (const auto& p : tetVerts) {
            double d = (p - center).magnitude();
            radius = std::max(radius, d);
        }

        // Add big enclosing tetrahedron
        double s = 5.0 * radius;
        tetVerts.push_back(Vec3(-s, 0.0, -s));
        tetVerts.push_back(Vec3( s, 0.0, -s));
        tetVerts.push_back(Vec3(0.0, s, s));
        tetVerts.push_back(Vec3(0.0, -s, s));

        // Create tetrahedra
        std::vector<uint32_t> faces = createTetIds(tetVerts, minQuality);

        int numTets = (int)faces.size() / 4;

        // Build output: tetrahedra indices only (points are unchanged)
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
    }, "points"_a, "minQuality"_a = 0.001,
       "Compute the Delaunay tetrahedralization of a point cloud. "
       "minQuality filters out degenerate tetrahedra (1.0 = perfect, 0.001 = default).");

    // Manifold Example: Create a simple cube
    m.def("create_cube", [](double x, double y, double z) {
        manifold::Manifold cube = manifold::Manifold::Cube({x, y, z});
        manifold::MeshGL64 mesh = cube.GetMeshGL64(); // Ensure the mesh is created
        return std::make_tuple(mesh.vertProperties, mesh.triVerts);
    }, "x"_a, "y"_a, "z"_a, "Create a cube with given dimensions");

    // Manifold Example: Create a simple cube
    m.def("create_sphere", [](double radius) {
        manifold::Manifold cube = manifold::Manifold::Sphere(radius);
        manifold::MeshGL64 mesh = cube.GetMeshGL64(); // Ensure the mesh is created
        return std::make_tuple(mesh.vertProperties, mesh.triVerts);
    }, "radius"_a, "Create a sphere with given radius");

    // Voro++ Example: Compute Voronoi diagram
    m.def("voronoi_3d", [](nb::ndarray<double, nb::shape<-1, 3>> points,
                           nb::ndarray<double, nb::shape<-1>> wts,
                           nb::ndarray<double, nb::shape<6>> bounds) {

        double min_x = bounds(0) - 0.1f, max_x = bounds(1) + 0.1f;
        double min_y = bounds(2) - 0.1f, max_y = bounds(3) + 0.1f;
        double min_z = bounds(4) - 0.1f, max_z = bounds(5) + 0.1f;
        if (points.shape(1) != 3) {
            throw std::invalid_argument("Points must be a 2D array with shape (N, 3)");
        }
        if (points.shape(0) == 0) {
            throw std::invalid_argument("Points array cannot be empty");
        }

        double V = (max_x - min_x) * (max_y - min_y) * (max_z - min_z);
        double Nthird = powf((double)points.shape(0) / V, 1.0f / 3.0f);
        voro::container_poly container(bounds(0), bounds(1), bounds(2), 
                                       bounds(3), bounds(4), bounds(5),
                                       std::round(Nthird * (max_x - min_x)),
                                       std::round(Nthird * (max_y - min_y)),
                                       std::round(Nthird * (max_z - min_z)),
                                       false, false, false, (int)points.shape(0));

        bool hasWeights = wts.size() == points.shape(0);
        for (int i = 0; i < points.shape(0); i++) {
            container.put(i, points(i, 0), points(i, 1), points(i, 2), hasWeights ? wts(i) : 1.0f);
        }

        // Voronoi Computation
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
                for (size_t i = 0; i < c.p; i++) {
                    verts.push_back(x + 0.5 * c.pts[(4 * i) + 0]);
                    verts.push_back(y + 0.5 * c.pts[(4 * i) + 1]);
                    verts.push_back(z + 0.5 * c.pts[(4 * i) + 2]);
                }
                cells.push_back(verts);
            }
        } while (vl.inc());

        return cells;//nb::ndarray<double, nb::numpy, nb::shape<-1, 3>>(verts).cast();;
    }, "points"_a, "wts"_a, "bounds"_a,
       "Compute Voronoi cell volumes for 3D points within given bounds [x_min, x_max, y_min, y_max, z_min, z_max]");

    m.def("getReflexEdges", [](
        nb::ndarray<  double, nb::shape<-1, 3>> points,
        nb::ndarray<uint32_t, nb::shape<-1, 3>> triangles) {
            std::vector<std::vector<size_t>> polygons = std::vector<std::vector<size_t>>();
            std::vector<geometrycentral::Vector3> vertexPositions = std::vector<geometrycentral::Vector3>();

            for(size_t i = 0; i < points.shape(0); i++) {
                auto vertex = geometrycentral::Vector3(
                    points(i, 0),
                    points(i, 1),
                    points(i, 2)
                );
                vertexPositions.push_back(vertex);
	        }

			for (size_t i = 0; i < triangles.shape(0); i++) {
                std::vector<size_t> polygon;
                polygon.push_back(triangles(i, 0));
                polygon.push_back(triangles(i, 1));
                polygon.push_back(triangles(i, 2));
                polygons.push_back(polygon);
            }

            // Create a Geometry Central half-edge mesh from the input triangles and points
            std::tuple<std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh>, 
                       std::unique_ptr<geometrycentral::surface::VertexPositionGeometry>> meshAndGeo =
                geometrycentral::surface::makeHalfedgeAndGeometry(polygons, vertexPositions);

			std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh = std::move(std::get<0>(meshAndGeo));
			std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry = std::move(std::get<1>(meshAndGeo));
            std::vector<std::tuple<std::array<size_t, 2>, std::array<size_t, 2>>> reflexEdges;
            // Identify reflex edges
            for (auto e : mesh->edges()) {
                auto he1 = e.halfedge();
                auto he2 = he1.twin();
                auto f1 = he1.face();
                auto f2 = he2.face();
                auto n1 = geometry->faceNormal(f1);
                auto n2 = geometry->faceNormal(f2);
                geometrycentral::Vector3 tangent =
                    geometrycentral::cross(n1,
                        geometry->vertexPositions[e.secondVertex().getIndex()] -
                        geometry->vertexPositions[e. firstVertex().getIndex()]);
                double tangentProjection = geometrycentral::dot(n2, tangent);
                //  If we've found a pair of reflex triangles, add them to the set
                if (tangentProjection > 0.000000001) {
                    size_t v1 = e.halfedge().vertex().getIndex();
                    size_t v2 = e.halfedge().twin().vertex().getIndex();
                    reflexEdges.push_back(std::make_tuple(std::array<size_t, 2> {v1, v2}, 
                                                          std::array<size_t, 2> { f1.getIndex(), f2.getIndex() }));
                }
            }
			return reflexEdges;
    }, "points"_a, "triangles"_a, "Identify reflex edges in a triangular mesh defined by input triangles");

    // Version info
    m.attr("__version__") = "0.1.0";
}
