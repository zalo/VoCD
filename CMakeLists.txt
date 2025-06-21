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

# Configure CDT as a library
# CDT is set up as an executable, so we need to create a library from its sources
file(GLOB CDT_SOURCES 
  extern/CDT/src/giftWrap.cpp
  extern/CDT/src/PLC.cpp
  extern/CDT/src/delaunay.cpp
  extern/CDT/src/main.cpp
)
add_library(CDT STATIC ${CDT_SOURCES})
target_include_directories(CDT PUBLIC 
  extern/CDT/src
  extern/CDT/include
)
target_compile_features(CDT PUBLIC cxx_std_20)
# Add CDT compile options
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(CDT PUBLIC /fp:strict /Oi "/D _CRT_SECURE_NO_WARNINGS" "/D USE_MAROTS_METHOD" "/D LGPL=ON")
else()
  target_compile_options(CDT PUBLIC -O2 -frounding-math "-DUSE_MAROTS_METHOD" "-DLGPL=ON")
endif()

# Configure Manifold
set(MANIFOLD_BUILD_SHARED OFF CACHE INTERNAL "")
set(MANIFOLD_PAR "NONE" CACHE INTERNAL "")
set(MANIFOLD_BUILD_TEST OFF CACHE INTERNAL "")
set(MANIFOLD_BUILD_SAMPLES OFF CACHE INTERNAL "")
add_subdirectory(extern/manifold)

## Configure Voro++
## Voro++ doesn't have a CMakeLists.txt in root, so we need to add it manually
## We need to manually specify sources and use a wrapper for v_base_wl.cc
#set(VORO_SOURCES
#  extern/voro/src/c_loops.cc
#  extern/voro/src/cell.cc
#  extern/voro/src/common.cc
#  extern/voro/src/container.cc
#  extern/voro/src/container_prd.cc
#  extern/voro/src/pre_container.cc
#  extern/voro/src/unitcell.cc
#  extern/voro/src/v_base.cc
#  extern/voro/src/v_compute.cc
#  extern/voro/src/voro++.cc
#  extern/voro/src/wall.cc
#  extern/voro/src/v_base_wl_wrapper.cc
#)
#add_library(voro STATIC ${VORO_SOURCES})
#target_include_directories(voro PUBLIC extern/voro/src)
#target_compile_features(voro PUBLIC cxx_std_11)
set(VORO_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE    OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(extern/voro)

# Create the Python module
nanobind_add_module(
  vocd
  NB_STATIC
  src/bindings.cpp
)

# Link libraries
target_link_libraries(vocd PRIVATE 
  CDT
  manifold
  voro++
)

# Set properties
target_compile_definitions(vocd PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install the Python module
install(TARGETS vocd LIBRARY DESTINATION vocd)