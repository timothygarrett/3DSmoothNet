cmake_minimum_required( VERSION 3.1 )

# Add C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create Project
project (3DSmoothNet)

add_executable(3DSmoothNet
  main.cpp
  core/core.cpp
  core/core.h
)

# Set StartUp Project (Option)
# (This setting is able to enable by using CMake 3.6.0 RC1 or later.)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

# Find Packages
find_package( PCL 1.8 REQUIRED )

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries(3DSmoothNet ${PCL_LIBRARIES} )
endif()

find_package(HDF5 COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})
target_link_libraries(3DSmoothNet ${HDF5_LIBRARIES})

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES( 3DSmoothNet ${Boost_LIBRARIES} )

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

