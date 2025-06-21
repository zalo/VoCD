#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

#include "delaunay.h"
#include "inputPLC.h"
#include "PLC.h"
#include <manifold/manifold.h>
#include "voro++.hh"

namespace nb = nanobind;
using namespace nb::literals;

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

        //// Build a structured PLC linked to the Delaunay tetrahedrization
        //PLCx Steiner_plc(*tin, plc.triangle_vertices.data(), plc.numTriangles());

        //// Recover segments by inserting Steiner points in both the PLC and the tetrahedrization
        //Steiner_plc.segmentRecovery_HSi(!verbose);

        //// Recover PLC faces by locally remeshing the tetrahedrization
        //bool sisMethodWorks = Steiner_plc.faceRecovery(!verbose);

        //// Mark the tets which are bounded by the PLC.
        //// If the PLC is not a valid polyhedron (i.e. it has odd-valency edges)
        //// all the tets but the ghosts are marked as "internal".
        //uint32_t num_inner_tets = (uint32_t)Steiner_plc.markInnerTets();

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

    // Manifold Example: Create a simple cube
    m.def("create_cube", [](double x, double y, double z) {
        manifold::Manifold cube = manifold::Manifold::Cube({x, y, z});
        manifold::MeshGL64 mesh = cube.GetMeshGL64(); // Ensure the mesh is created
        return std::make_tuple(mesh.vertProperties, mesh.triVerts);
    }, "x"_a, "y"_a, "z"_a, "Create a cube with given dimensions");

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

    // Version info
    m.attr("__version__") = "0.1.0";
}
