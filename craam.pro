TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    test/test.cpp

SUBDIRS += \
    craam.pro

DISTFILES += \
    craam.pro.user

INCLUDEPATH += $$PWD
INCLUDEPATH += $$PWD/include

HEADERS += \
    craam/Action.hpp \
    craam/config.hpp \
    craam/definitions.hpp \
    craam/ImMDP.hpp \
    craam/modeltools.hpp \
    craam/RMDP.hpp \
    craam/Samples.hpp \
    craam/Simulation.hpp \
    craam/State.hpp \
    craam/Transition.hpp \
    craam/algorithms/occupancies.hpp \
    craam/algorithms/robust_values.hpp \
    craam/algorithms/values.hpp \
    test/core_tests.hpp \
    test/implementable_tests.hpp \
    test/simulation_tests.hpp \
    craam/fastopt.hpp

QMAKE_CXXFLAGS += -fopenmp

LIBS += -fopenmp \
        -lboost_unit_test_framework
