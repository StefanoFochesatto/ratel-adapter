cmake_minimum_required(VERSION 3.13.4)

project(ratel-adapter VERSION 0.1.0 LANGUAGES C CXX)

# C++ is needed for preCICE
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)

# Set default build type to Debug for development
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Find dependencies
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)

# Find PETSc using pkg-config
set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_check_modules(PETSC REQUIRED PETSc>=3.24)

message(STATUS "PETSc found:")
message(STATUS "  CFLAGS: ${PETSC_CFLAGS}")
message(STATUS "  LIBS: ${PETSC_LIBRARIES}")
message(STATUS "  LDFLAGS: ${PETSC_LDFLAGS}")
message(STATUS "  INCLUDE: ${PETSC_INCLUDE_DIRS}")

# Find preCICE using pkg-config
pkg_check_modules(PRECICE REQUIRED libprecice>=3.0)

message(STATUS "preCICE found:")
message(STATUS "  CFLAGS: ${PRECICE_CFLAGS}")
message(STATUS "  LIBS: ${PRECICE_LIBRARIES}")
message(STATUS "  INCLUDE: ${PRECICE_INCLUDE_DIRS}")

# Find Ratel
find_path(RATEL_INCLUDE_DIR
    NAMES ratel.h
    PATHS 
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../ratel/include
        /usr/local/include
        $ENV{RATEL_DIR}/include
)

find_library(RATEL_LIBRARY
    NAMES ratel
    PATHS
        ${CMAKE_SOURCE_DIR}/../lib
        ${CMAKE_SOURCE_DIR}/../ratel/lib
        /usr/local/lib
        $ENV{RATEL_DIR}/lib
)

# Check if we found Ratel headers
if(NOT RATEL_INCLUDE_DIR)
    if(EXISTS ${CMAKE_SOURCE_DIR}/../include/ratel.h)
        set(RATEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../include)
        message(STATUS "Found Ratel headers in parent directory")
    elseif(EXISTS ${CMAKE_SOURCE_DIR}/../ratel/include/ratel.h)
        set(RATEL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../ratel/include)
        message(STATUS "Found Ratel headers in ../ratel/include")
    else()
        message(FATAL_ERROR "Could not find Ratel headers. Set RATEL_INCLUDE_DIR manually.")
    endif()
endif()

message(STATUS "Ratel include dir: ${RATEL_INCLUDE_DIR}")
message(STATUS "Ratel library: ${RATEL_LIBRARY}")

# Library target
add_library(ratel-adapter SHARED
    src/adapter.c
    src/mesh_extraction.c
    src/data_mapping.c
)

target_include_directories(ratel-adapter
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PRECICE_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${PETSC_INCLUDE_DIRS}
        ${RATEL_INCLUDE_DIR}
)

target_compile_options(ratel-adapter PRIVATE ${PETSC_CFLAGS_OTHER} ${PRECICE_CFLAGS_OTHER})

target_link_libraries(ratel-adapter
    PUBLIC
        ${PRECICE_LIBRARIES}
        MPI::MPI_C
        MPI::MPI_CXX
    PRIVATE
        ${PETSC_LIBRARIES}
)

# Set library version
set_target_properties(ratel-adapter PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME ratel-adapter
)

# Set output directories for examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/examples)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/examples)

# Build Examples
if(EXISTS ${CMAKE_SOURCE_DIR}/examples/ex02-quasistatic-precice.c)
    add_executable(ex02-quasistatic-precice examples/ex02-quasistatic-precice.c)
    
    target_include_directories(ex02-quasistatic-precice PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${RATEL_INCLUDE_DIR}
        ${PETSC_INCLUDE_DIRS}
        ${PRECICE_INCLUDE_DIRS}
    )
    
    # Link in correct order: executable -> ratel-adapter -> ratel -> petsc -> precice -> mpi
    target_link_libraries(ex02-quasistatic-precice 
        ratel-adapter
        ${RATEL_LIBRARY}
        ${PETSC_LDFLAGS}
        ${PRECICE_LIBRARIES}
        MPI::MPI_C
        MPI::MPI_CXX
    )
    
    # Set debug flags for development
    target_compile_options(ex02-quasistatic-precice PRIVATE -g -O0)
endif()

# Build ex03-dynamic-precice example
if(EXISTS ${CMAKE_SOURCE_DIR}/examples/ex03-dynamic-precice.c)
    add_executable(ex03-dynamic-precice examples/ex03-dynamic-precice.c)

    target_include_directories(ex03-dynamic-precice PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${RATEL_INCLUDE_DIR}
        ${PETSC_INCLUDE_DIRS}
        ${PRECICE_INCLUDE_DIRS}
    )

    # Link in correct order: executable -> ratel-adapter -> ratel -> petsc -> precice -> mpi
    target_link_libraries(ex03-dynamic-precice
        ratel-adapter
        ${RATEL_LIBRARY}
        ${PETSC_LDFLAGS}
        ${PRECICE_LIBRARIES}
        MPI::MPI_C
        MPI::MPI_CXX
    )

    # Set debug flags for development
    target_compile_options(ex03-dynamic-precice PRIVATE -g -O0)
endif()

# Install rules
include(GNUInstallDirs)

install(TARGETS ratel-adapter
    EXPORT ratel-adapter-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT ratel-adapter-targets
    FILE ratel-adapter-targets.cmake
    NAMESPACE ratel-adapter::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ratel-adapter
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ratel-adapter-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ratel-adapter-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ratel-adapter
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library output: ${CMAKE_BINARY_DIR}")
message(STATUS "  Examples output: ${CMAKE_BINARY_DIR}/examples")
message(STATUS "")
