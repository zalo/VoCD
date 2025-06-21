# Voronoi Convex Decomposition

A library for computing the convex decomposition of meshes via Constrained Delaunay Tetrahedralization and Voronoi Diagrams. (WIP)

# Algorithm

1. Tetrahedralize the Mesh via CDT
2. Find all the tetrahedra adjacent to reflex edges (edges which have a concave angle)
3. Place a Voronoi Cell in the circumcenter of each of these tetrahedra (with the radius set to the circumradius)
4. Compute the voronoi cells and intersect them with the original mesh via manifold
5. Fin!

## Dependencies

This project includes the following libraries as git submodules:
- [nanobind](https://github.com/wjakob/nanobind) - Fast Python bindings
- [CDT](https://github.com/MarcoAttene/CDT) - Constrained Delaunay Triangulation
- [Manifold](https://github.com/elalish/manifold) - Geometry library for topological manifolds
- [Voro++](https://github.com/chr1shr/voro) - 3D Voronoi cell software library

## Installation

### From PyPI (when published)
```bash
pip install vocd
```

### From source
```bash
# Clone with submodules
git clone --recursive git@github.com:zalo/VoCD.git
cd VoCD

# Install with pip
pip install .

# Or install in development mode
pip install -e .
```

### Building wheels with cibuildwheel
```bash
# Install cibuildwheel
pip install cibuildwheel

# Build wheels for current platform
cibuildwheel --platform auto

# Or use GitHub Actions (see .github/workflows/wheels.yml)
```

## Usage (TODO)