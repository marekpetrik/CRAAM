cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

set(VERSION "0.3.0")
project (CRAAM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

find_package(Boost COMPONENTS unit_test_framework)
find_package(Doxygen)

#set(CMAKE_VERBOSE_MAKEFILE on)
option (BUILD_SIMULATION "Build simulation functionality (requires C++14)" ON)
option (BUILD_TEST "Build tests (requires Boost)" ON)
option (BUILD_DOCUMENTATION "Build source code documentation" ${DOXYGEN_FOUND})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories ("include")

set (SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/Action.cpp  
         ${CMAKE_CURRENT_SOURCE_DIR}/src/definitions.cpp  
         ${CMAKE_CURRENT_SOURCE_DIR}/src/RMDP.cpp  
         ${CMAKE_CURRENT_SOURCE_DIR}/src/State.cpp  
         ${CMAKE_CURRENT_SOURCE_DIR}/src/Transition.cpp)
set (TSTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (BUILD_SIMULATION)
    set (SRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/src/Simulation.cpp)
    set (TSTS ${TSTS} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_simulation.cpp)
    set (CMAKE_CXX_STANDARD 14)
endif (BUILD_SIMULATION)

# **** RELEASE VERSION OF THE LIBRARY ****
add_library (craam ${SRCS} )
set_target_properties(craam PROPERTIES CMAKE_BUILD_TYPE Release)

# **** DEBUG VERSION OF THE LIBRARY ****
add_library (craam_debug ${SRCS} )
set_target_properties(craam_debug PROPERTIES CMAKE_BUILD_TYPE Debug)

# **** TEST ****
if (BUILD_TEST)
    add_executable (unit_tests_debug ${TSTS} )
    target_link_libraries(unit_tests_debug ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} craam_debug)
    
    add_executable (unit_tests ${TSTS} )
    target_link_libraries(unit_tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} craam)
    
    add_custom_target (testit   COMMAND unit_tests --show_progress --detect_memory_leaks --detect_fp_exceptions
                                COMMAND unit_tests_debug --show_progress --detect_memory_leaks --detect_fp_exceptions
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
                        COMMENT "Running unit tests")
    
    #enable_testing()
    #add_test(UnitTests bin/unit_tests_debug)
    #add_test(UnitTests bin/unit_tests)
endif (BUILD_TEST)

# **** BENCHMARK ****
# builds, downloads and runs a benchmark

# **** DOCUMENTATION ****
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Needs Doxygen to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen.config.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doc/doxyfile.config)
    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    #install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
