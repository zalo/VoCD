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

# Find Python
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
               OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Scikit-build-core sets these values
if (DEFINED SKBUILD)
  # Add nanobind
  add_subdirectory(extern/nanobind)
else()
  # Import nanobind through CMake's find_package mechanism
  set(nanobind_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/nanobind/cmake")
  find_package(nanobind CONFIG REQUIRED)
endif()
# Add the include directories for nanobind
include_directories(${nanobind_INCLUDE_DIRS})

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
set(MANIFOLD_BUILD_SHARED  OFF    CACHE INTERNAL "")
set(BUILD_SHARED_LIBS      OFF    CACHE BOOL "" FORCE)
set(MANIFOLD_PAR           "NONE" CACHE INTERNAL "")
set(MANIFOLD_BUILD_TEST    OFF    CACHE INTERNAL "")
set(MANIFOLD_BUILD_SAMPLES OFF    CACHE INTERNAL "")
add_subdirectory(extern/manifold)

# Configure Voro++
set(VORO_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(VORO_BUILD_CMD_LINE    OFF CACHE BOOL "" FORCE)
set(VORO_ENABLE_DOXYGEN    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(extern/voro)

# Create the Python module
nanobind_add_module(vocd_ext STABLE_ABI LTO src/vocd_ext.cpp)

# Link libraries
target_link_libraries(vocd_ext PRIVATE 
  CDT
  manifold
  voro++
)

# Set properties
target_compile_definitions(vocd_ext PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Install the Python module
install(TARGETS vocd_ext LIBRARY DESTINATION vocd)
