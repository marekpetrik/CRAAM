cmake_minimum_required (VERSION 2.6)
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

project (CRAAM)

option (BUILD_SIMULATION "Build simulation functionality" ON)
option (BUILD_TEST "Build tests" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories ("include")

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

set (SRCS src/Action.cpp  
         src/definitions.cpp  
         src/RMDP.cpp  
         src/State.cpp  
         src/Transition.cpp)
set (TSTS test/test.cpp)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

if (BUILD_SIMULATION)
    set (SRCS ${SRCS} src/Simulation.cpp)
    set (TSTS ${TSTS} test/test_simulation.cpp)
    set (CMAKE_CXX_STANDARD 14)
endif (BUILD_SIMULATION)

add_library (craam ${SRCS} )
#install (TARGETS craam DESTINATION bin)

if (BUILD_TEST)
    add_executable (unit_tests ${TSTS} )
    target_link_libraries(unit_tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} craam)
    enable_testing()
    add_test(UnitTests bin/unit_tests)
endif (BUILD_TEST)
