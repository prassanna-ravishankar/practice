#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace boost::python;

class EigenWrap
{
public:
    EigenWrap(){}
    void set_array(PyObject* numpy_array);
    PyObject* eigen_values();

private:
    MatrixXd mat;
};

void EigenWrap::set_array(PyObject* numpy_array)
{
    npy_intp *shape = PyArray_DIMS(numpy_array);
    mat = Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(static_cast<double *> PyArray_DATA(numpy_array),shape[0], shape[1]);
}

PyObject* EigenWrap::eigen_values()
{
    SelfAdjointEigenSolver<MatrixXd> solver(mat);
    VectorXd eigen_values = solver.eigenvalues();
    npy_intp shape[] = {mat.rows()};
    PyObject *ret = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, eigen_values.data());
    return ret;
}

BOOST_PYTHON_MODULE(_EigenWrap)
{
    class_<EigenWrap>("EigenWrap")
        .def("set_array", &EigenWrap::set_array)
        .def("eigen_values", &EigenWrap::eigen_values)
    ;
}
