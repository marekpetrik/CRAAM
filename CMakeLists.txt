cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

set(VERSION "0.3.0")
project (CRAAM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(OpenMP)
# CMake does not detect header-only packages. Also needs uBlas and format
find_package(Boost COMPONENTS unit_test_framework ) 
find_package(Doxygen)

if(${Boost_FOUND} LESS 1)
    message(WARNING "Unit tests require Boost." )
endif()

# Prints all variables (for debugging)
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


#set(CMAKE_VERBOSE_MAKEFILE on) - just run make 
option (BUILD_TESTS "Build tests (requires Boost)" ${Boost_FOUND})
option (BUILD_DOCUMENTATION "Build source code documentation" ${DOXYGEN_FOUND})
option (BUILD_SIMULATION "Build simulation functionality" ON)
option (BUILD_BENCHMARK "Build benchmaking functionality" ON)
option (BUILD_IMPLEMENTABLE "Build tools for solving with implementability constraints" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# use position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include_directories ("include" "ext_include" )

# **** SOURCE FILES ****
set (SRCS 
          ${CMAKE_CURRENT_SOURCE_DIR}/src/Action.cpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Action.hpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/src/definitions.cpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/include/definitions.hpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/src/RMDP.cpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/include/RMDP.hpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/src/State.cpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/include/State.hpp  
          ${CMAKE_CURRENT_SOURCE_DIR}/src/Transition.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Transition.hpp
          )
set (TSTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
set (DEV ${CMAKE_CURRENT_SOURCE_DIR}/test/dev.cpp)
set (BENCH ${CMAKE_CURRENT_SOURCE_DIR}/test/benchmark.cpp)

# whether to build the simulation component of the library
if (BUILD_SIMULATION)
    set (SRCS ${SRCS} 
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/Simulation.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/Simulation.hpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/Samples.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/Samples.hpp
                    )
    set (TSTS ${TSTS} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_simulation.cpp)
endif (BUILD_SIMULATION)

# whether to build the implementable component of the library
if (BUILD_IMPLEMENTABLE)
    set (SRCS ${SRCS} 
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/ImMDP.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/ImMDP.hpp)
    set (TSTS ${TSTS} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_implementable.cpp)
endif (BUILD_IMPLEMENTABLE)

# use OpenMP if possible
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set gcc-specific parameters
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -rdynamic") # -D_GLIBCXX_DEBUG -- incompatible with boost :(
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif()

#add_library (craam SHARED ${SRCS} )
add_library (craam STATIC ${SRCS} )
#target_link_libraries(craam ${ARMADILLO_LIBRARIES} )

add_executable (develop ${DEV})
target_link_libraries(develop craam)

# **** TEST ****
if (BUILD_TESTS)
    add_executable (unit_tests ${TSTS} )
    target_link_libraries(unit_tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} craam)
    
    add_custom_target (testit   COMMAND unit_tests --show_progress --detect_memory_leaks --detect_fp_exceptions
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
                        COMMENT "Running unit tests")
endif (BUILD_TESTS)

# **** BENCHMARK ****
# builds, downloads and runs a benchmark
if (BUILD_BENCHMARK)
    add_executable(benchmark EXCLUDE_FROM_ALL ${BENCH} )
    target_link_libraries(benchmark ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} craam)
endif()

# **** DOCUMENTATION ****
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Needs Doxygen to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen.config.in)
    set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxyfile.config)
    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    #install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
