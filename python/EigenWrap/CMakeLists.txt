cmake_minimum_required(VERSION 2.6)

set(EIGEN_INCLUDE_DIR "")
set(PYTHON_INCLUDE_DIR "")
set(BOOST_INCLUDE_DIR "")

set(PYTHON_LIB_DIR "")
set(BOOST_LIB_DIR "")

INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})

link_directories(${PYTHON_LIB_DIR} ${BOOST_LIB_DIR})

set(LIBS boost_python python2.7)
ADD_LIBRARY(pyeigen wrapper_example.cpp)
TARGET_LINK_LIBRARIES(pyeigen ${LIBS})