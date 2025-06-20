#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

#include <CDT.h>
#include <manifold/manifold.h>
#include <voro++.hh>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(vocd, m) {
    m.doc() = "Geometry tools using CDT, Manifold, and Voro++";

    // CDT Example: 2D Constrained Delaunay Triangulation
    m.def("triangulate_2d", [](nb::ndarray<double, nb::shape<-1, 2>> points) {
        CDT::Triangulation<double> cdt;
        
        // Add vertices
        std::vector<CDT::V2d<double>> vertices;
        auto view = points.unchecked<2>();
        for (size_t i = 0; i < view.shape(0); ++i) {
            vertices.push_back({view(i, 0), view(i, 1)});
        }
        cdt.insertVertices(vertices);
        
        // Triangulate
        cdt.eraseSuperTriangle();
        
        // Extract triangles
        std::vector<std::array<int, 3>> triangles;
        for (const auto& tri : cdt.triangles) {
            triangles.push_back({
                static_cast<int>(tri.vertices[0]),
                static_cast<int>(tri.vertices[1]),
                static_cast<int>(tri.vertices[2])
            });
        }
        
        return triangles;
    }, "points"_a, "Perform 2D Delaunay triangulation on a set of points");

    // Manifold Example: Create a simple cube
    m.def("create_cube", [](double x, double y, double z) {
        return manifold::Manifold::Cube({x, y, z});
    }, "x"_a=1.0, "y"_a=1.0, "z"_a=1.0, "Create a cube with given dimensions");

    // Voro++ Example: Compute Voronoi diagram
    m.def("voronoi_3d", [](nb::ndarray<double, nb::shape<-1, 3>> points, 
                          std::array<double, 6> bounds) {
        voro::container con(bounds[0], bounds[1], bounds[2], 
                           bounds[3], bounds[4], bounds[5],
                           10, 10, 10, false, false, false, 8);
        
        // Add particles
        auto view = points.unchecked<2>();
        for (size_t i = 0; i < view.shape(0); ++i) {
            con.put(i, view(i, 0), view(i, 1), view(i, 2));
        }
        
        // Compute volumes
        std::vector<double> volumes;
        voro::c_loop_all vl(con);
        voro::voronoicell_neighbor c;
        if (vl.start()) {
            do {
                if (con.compute_cell(c, vl)) {
                    volumes.push_back(c.volume());
                }
            } while (vl.inc());
        }
        
        return volumes;
    }, "points"_a, "bounds"_a, 
       "Compute Voronoi cell volumes for 3D points within given bounds [x_min, x_max, y_min, y_max, z_min, z_max]");

    // Version info
    m.attr("__version__") = "0.1.0";
}