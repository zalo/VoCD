cmake_minimum_required(VERSION 3.15...3.27)
project(vocd LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Scikit-build-core sets these values
if (DEFINED SKBUILD)
  set(PYTHON_EXECUTABLE "${SKBUILD_PYTHON_EXECUTABLE}")
  set(PYTHON_INCLUDE_DIR "${SKBUILD_PYTHON_INCLUDE_DIR}")
  set(PYTHON_LIBRARY "${SKBUILD_PYTHON_LIBRARY}")
  set(CMAKE_INSTALL_PREFIX "${SKBUILD_PLATLIB_DIR}")
endif()

# Add nanobind
add_subdirectory(extern/nanobind)

# Find Python and nanobind
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
nanobind_find_python()

# Configure CDT
add_subdirectory(extern/CDT)

# Configure Manifold
set(MANIFOLD_BUILD_SHARED OFF CACHE INTERNAL "")
set(MANIFOLD_PAR "NONE" CACHE INTERNAL "")
set(MANIFOLD_BUILD_TEST OFF CACHE INTERNAL "")
set(MANIFOLD_BUILD_SAMPLES OFF CACHE INTERNAL "")
add_subdirectory(extern/manifold)

# Configure Voro++
# Voro++ doesn't have a CMakeLists.txt in root, so we need to add it manually
file(GLOB VORO_SOURCES extern/voro/src/*.cc)
add_library(voro STATIC ${VORO_SOURCES})
target_include_directories(voro PUBLIC extern/voro/src)
target_compile_features(voro PUBLIC cxx_std_11)

# Create the Python module
nanobind_add_module(
  vocd
  NB_STATIC
  src/bindings.cpp
)

# Link libraries
target_link_libraries(vocd PRIVATE 
  CDT::CDT
  manifold
  voro
)

# Set properties
target_compile_definitions(vocd PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install the Python module
install(TARGETS vocd LIBRARY DESTINATION vocd)