# Nanobind Template Project

A template CMake project that demonstrates how to create Python bindings using nanobind with multiple computational geometry libraries.

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
git clone --recursive <your-repo-url>
cd nanobind-template-project

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

## Usage

```python
import vocd

# 2D triangulation
import numpy as np
points = np.random.rand(10, 2)
triangles = vocd.triangulate_2d(points)

# Create a cube using Manifold
cube = vocd.create_cube(2.0, 3.0, 4.0)

# Compute 3D Voronoi cell volumes
points_3d = np.random.rand(20, 3) * 10
bounds = [0, 10, 0, 10, 0, 10]
volumes = vocd.voronoi_3d(points_3d, bounds)
```

## Development

To add more bindings, edit `src/bindings.cpp` and rebuild the project.