cmake_minimum_required (VERSION 3.8)
project (colza-ascent-pipeline C CXX)
set (CMAKE_CXX_STANDARD 14)

# link shared lib with full rpath
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# setup cache variables for ccmake
if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build." FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")

# find dependencies
find_package (colza REQUIRED)
find_package (MPI REQUIRED)
find_package (spdlog REQUIRED)
find_package (Ascent REQUIRED)
find_package (PkgConfig REQUIRED)
pkg_check_modules (MONA_MPI REQUIRED IMPORTED_TARGET mona-mpi)

add_library (colza-ascent-pipeline SHARED src/colza-ascent-pipeline.cpp)
target_link_libraries (colza-ascent-pipeline MPI::MPI_C colza-server spdlog::spdlog ascent::ascent_mpi PkgConfig::MONA_MPI)

install (TARGETS colza-ascent-pipeline
         EXPORT colza-ascent-pipeline
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib)
