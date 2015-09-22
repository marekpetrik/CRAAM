cmake_minimum_required (VERSION 2.6)
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)

project (CRAAM)

option (BUILD_SIMULATION "Build simulation functionality" ON)
option (BUILD_TEST "Build tests" ON)

include_directories ("include")

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

set (SRCS src/Action.cpp  
         src/definitions.cpp  
         src/RMDP.cpp  
         src/State.cpp  
         src/Transition.cpp)
set (TSTS test/test.cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED)

if (BUILD_SIMULATION)
    set (SRCS ${SRCS} src/Simulation.cpp)
    set (TSTS ${TSTS} test/test_simulation.cpp)
    set (CMAKE_CXX_STANDARD 14)
endif (BUILD_SIMULATION)

if (BUILD_TEST)
    add_executable (basic_test ${TSTS} ${SRCS})
    target_link_libraries(basic_test ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    enable_testing()
    add_test(basic_test basic_test)
endif (BUILD_TEST)
