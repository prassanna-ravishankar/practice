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
    void set_array(PyArrayObject* numpy_array);
    PyObject* eigen_values();

private:
  //    Map<Matrix<double, Dynamic, Dynamic, RowMajor> > mat;
  MatrixXd mat;
};

void EigenWrap::set_array(PyArrayObject* numpy_array)
{
    npy_intp *shape = PyArray_DIMS(numpy_array);
    mat = Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(static_cast<double *> PyArray_DATA(numpy_array),shape[0], shape[1]);
}

PyObject* EigenWrap::eigen_values()
{
    SelfAdjointEigenSolver<MatrixXd> solver(mat);
    VectorXd eigen_values = solver.eigenvalues();
    npy_intp shape[] = {mat.rows()};
    // PyObject *ret = PyArray_SimpleNewFromData(1, shape, NPY_DOUBLE, eigen_values.data());
    PyObject *ret = PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    for(int i = 0; i < mat.rows(); ++i)
      ((double *)PyArray_GETPTR1(ret, i))[0] = eigen_values(i);

    return ret;
}

void* extract_pyarray(PyObject* x)
{
	return x;
}


BOOST_PYTHON_MODULE(_EigenWrap)
{
  boost::python::converter::registry::insert(
	    &extract_pyarray, boost::python::type_id<PyArrayObject>());
  
    class_<EigenWrap>("EigenWrap")
        .def("set_array", &EigenWrap::set_array)
        .def("eigen_values", &EigenWrap::eigen_values)
    ;

    import_array();
}
